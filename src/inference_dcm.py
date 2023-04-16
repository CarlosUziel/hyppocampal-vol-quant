"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pydicom
import torch
from PIL import Image, ImageDraw, ImageFont

from inference.UNetInferenceAgent import UNetInferenceAgent


def get_series_for_inference(study_path: Path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        study_path: Location of the DICOM files

    Returns:
        Numpy array representing the series
    """
    # 1. Get series representing the hippocampus area

    series_list = [pydicom.dcmread(dcm_file) for dcm_file in study_path.glob("*.dcm")]
    series_for_inference = [
        series
        for series in series_list
        if "hippocrop" in series.SeriesDescription.lower()
    ]

    # 2. Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []
    else:
        return series_for_inference


def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """

    # In the real world you would do a lot of validation here
    slices = [
        np.flip(dcm.pixel_array).T
        for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)
    ]

    # Make sure that you have correctly constructed the volume from your axial slices!
    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)


def get_predicted_volumes(pred: np.array) -> Dict[str, int]:
    """Gets volumes of two hippocampal structures from the predicted array

    Args:
        pred: Array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """
    volume_ant = np.sum(pred == 1)
    volume_post = np.sum(pred == 2)

    return {
        "anterior": volume_ant,
        "posterior": volume_post,
        "total": volume_ant + volume_post,
    }


def create_report(
    inference: Dict[str, int], header: Any, orig_vol: np.array, pred_vol: np.array
):
    """Generates an image with inference report. The code below uses PIL image library
        to compose an RGB image that will go into the report. A standard way of storing
        measurement data in DICOM archives is creating such report and sending them on
        as Secondary Capture IODs
        (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html).
        Essentially, our report is just a standard RGB image, with some metadata, packed
        into DICOM format.

    Args:
        inference: Dict containing anterior, posterior and full volume values.
        header: DICOM header.
        orig_vol: Original volume.
        pred_vol: Predicted label.

    Returns:
        PIL image
    """
    # 1. Get base empty image
    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    # 2. Define header and body fonts
    font_file = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("assets")
        .joinpath("Roboto-Regular.ttf")
    )
    header_font = ImageFont.truetype(str(font_file), size=40)
    main_font = ImageFont.truetype(str(font_file), size=20)

    # 3. Write patient and predictions information
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text(
        (10, 90),
        (
            f"Patient ID: {header.PatientID} (Scan {header.InstanceNumber})\n"
            f"Volume dimensions: {orig_vol.shape}\n"
            f"Measured anterior hippocampus volume: {inference['anterior']}\n"
            f"Measured posterior hippocampus volume: {inference['posterior']}\n"
            f"Measured total hippocampus volume: {inference['total']}"
        ),
        (255, 255, 255),
        font=main_font,
    )

    # 4. Get slice with biggest mask (highest number of detections)
    slices_mask_area = [
        (i, np.sum(np.isin(vol_slice, (1, 2)))) for i, vol_slice in enumerate(pred_vol)
    ]
    idx, area = sorted(slices_mask_area, key=lambda x: x[1], reverse=True)[0]
    orig_slice, pred_slice = (orig_vol[idx, :, :], pred_vol[idx, :, :])

    draw.multiline_text(
        (10, 250),
        f"Slice {idx} with maximum area predicted ({area} mm3)",
        (255, 255, 255),
        font=main_font,
    )

    # 5. Generate pillow images for the report
    # Numpy array needs to flipped, transposed and normalized to a matrix of values in
    # the range of [0..255]
    orig_slice_img = (
        Image.fromarray(
            np.flip((orig_slice / np.max(orig_slice)) * 0xFF).T.astype(np.uint8),
            mode="L",
        )
        .convert("RGBA")
        .resize((600, 600))
    )
    mask_slice_img = Image.fromarray(
        np.flip((pred_slice / np.max(pred_slice)) * 0xFF).T.astype(np.uint8),
        mode="L",
    ).resize((600, 600))
    green = Image.new("RGB", orig_slice_img.size, (0, 255, 0))

    # Paste the PIL image into our main report image object (pimg)
    pimg.paste(orig_slice_img, box=(10, 300))
    pimg.paste(green, box=(10, 300), mask=mask_slice_img)

    return pimg


def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """

    # Code below creates a DICOM Secondary Capture instance that will be correctly
    # interpreted by most imaging viewers including our OHIF
    # The code here is complete as it is unlikely that as a data scientist you will
    # have to dive that deep into generating DICOMs. However, if you still want to understand
    # the subject, there are some suggestions below

    # Set up DICOM metadata fields. Most of them will be the same as original file header
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # STAND OUT SUGGESTION:
    # If you want to understand better the generation of valid DICOM, remove everything below
    # and try writing your own DICOM generation code from scratch.
    # Refer to this part of the standard to see what are the requirements for the valid
    # Secondary Capture IOD: http://dicom.nema.org/medical/dicom/2019e/output/html/part03.html#sect_A.8
    # The Modules table (A.8-1) contains a list of modules with a notice which ones are mandatory (M)
    # and which ones are conditional (C) and which ones are user-optional (U)
    # Note that we are building an RGB image which would have three 8-bit samples per pixel
    # Also note that writing code that generates valid DICOM has a very calming effect
    # on mind and body :)

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT"  # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = (  # We are deriving this image from patient data
        r"DERIVED\PRIMARY\AXIAL"
    )
    out.SamplesPerPixel = 3  # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0  # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    out.BitsAllocated = 8  # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)


def os_command(command):
    # Comment this if running under Windows
    sp = subprocess.Popen(["/bin/bash", "-i", "-c", command])
    sp.communicate()

    # Uncomment this if running under Windows
    # os.system(command)


if __name__ == "__main__":
    # 1. Get directory of DICOM studies
    if len(sys.argv) != 2:
        print(
            "You should supply one command line argument pointing to the routing"
            " folder. Exiting."
        )
        sys.exit()
    else:
        study_dir = Path(sys.argv[1])

    # 2. Get the HyppoCrop series
    series_dir = sorted(study_dir.glob("*HCropVolume"))[0]

    # 4. Look for suitable series and build 3D volume
    print(f"Looking for images to run inference on in directory {series_dir}...")
    volume, header = load_dicom_volume_as_numpy_from_list(
        get_series_for_inference(series_dir)
    )
    print(f"Found images of {volume.shape[0]} sagittal slices")

    # 5. Run inference on volume
    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        parameter_file_path=(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("data")
            .joinpath("output")
            .joinpath("section_2")
            .joinpath("test_results")
            .joinpath("2023-04-16_0803_basic_unet")
            .joinpath("model.pth.bz2")
        ),
    )

    # 6. Run inference
    pred_label = (
        inference_agent.single_volume_inference_unpadded(np.array(volume)).cpu().numpy()
    )
    pred_volumes = get_predicted_volumes(pred_label)

    # 7. Create and save the report
    print("Creating and pushing report...")
    report_save_path = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("data")
        .joinpath("output")
        .joinpath("section_3")
        .joinpath("report.dcm")
    )
    report_img = create_report(pred_volumes, header, volume, pred_label)
    report_img.save(report_save_path.parent.joinpath("report_image.png"))
    save_report_as_dcm(header, report_img, report_save_path)

    # 8. Send report to our storage archive
    os_command(f'storescu 127.0.0.1 4242 -v -aec REPORT +r +sd "{report_save_path}"')

    # 9. Final report
    print(
        (
            f"Inference successful on {header['SOPInstanceUID'].value}, out:"
            f" {pred_label.shape}"
        ),
        f"volume ant: {pred_volumes['anterior']}, ",
        (
            f"volume post: {pred_volumes['posterior']}, total volume:"
            f" {pred_volumes['total']}"
        ),
    )
