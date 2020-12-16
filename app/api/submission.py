from hashlib import sha512
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from requests import get

from app.utils.complexity.squad_score import squad_score, scaler
from app.utils.img_processing.google_api import GoogleAPI, NoTextFoundException
from app.api.models import Submission, ImageSubmission

import cv2
from skimage.transform import rotate
from deskew import determine_skew
import numpy as np

# global variables and services
router = APIRouter()
log = logging.getLogger(__name__)
vision = GoogleAPI()


@router.post("/submission/text")
async def submission_text(sub: Submission):
    """Takes a Submission Object and calls the Google Vision API to text annotate
    the passed s3 link, then passes those concatenated transcriptions to the SquadScore
    method, returns:
    Arguments:
    ---
    `sub`: Submission - Submission object **see `help(Submission)` for more info**
    Returns:
    ---
    ```
    {"SubmissionID": int, "IsFlagged": boolean,"LowConfidence": boolean, "Complexity": int}
    ```
    """
    transcriptions = ""
    confidence_flags = []
    # unpack links for files in submission object
    for page_num in sub.Pages:
        # re-init the sha algorithm every file that is processed
        hash = sha512()
        # fetch file from s3 bucket
        r = get(sub.Pages[page_num]["URL"])
        # update the hash with the file's content
        hash.update(r.content)
        try:
            # assert that the hash is the same as the one passed with the file
            # link
            assert hash.hexdigest() == sub.Pages[page_num]["Checksum"]
        except AssertionError:
            # return some useful information about the error including what
            # caused it and the file affected
            return JSONResponse(
                status_code=422,
                content={"ERROR": "BAD CHECKSUM", "file": sub.Pages[page_num]},
            )
        # now that the image passed the above grab just what i need
        r = get(sub.Pages[page_num]["URL"], stream=True).raw 
        # convert to array for transformation
        img = np.asarray(bytearray(r.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # gray scale it
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, im_gray_th_otsu = cv2.threshold(gray, 128, 192, cv2.THRESH_OTSU)
        im_gray_th_otsu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 211, 11,)
        # rotate it
        angle = determine_skew(im_gray_th_otsu)
        rotated = rotate(im_gray_th_otsu, angle, resize=True) * 255
        # change to bytes for apu to work
        success, encoded_image = cv2.imencode('.jpg', rotated)
        content = encoded_image.tobytes()
        # unpack response from GoogleAPI
        conf_flag, flagged, trans = await vision.transcribe(content)
        # concat transcriptions togeather
        transcriptions += trans + "\n"
        # add page to list of confidence flags
        confidence_flags.append(conf_flag)
    # score the transcription using SquadScore algorithm
    score = await squad_score(transcriptions, scaler)

    # return the complexity score to the web team with the SubmissionID
    return JSONResponse(
        status_code=200,
        content={
            "SubmissionID": sub.SubmissionID,
            "IsFlagged": flagged,
            "LowConfidence": True in confidence_flags,
            "Complexity": score,
        },
    )


@router.post("/submission/illustration")
async def submission_illustration(sub: ImageSubmission):
    """Function that checks the illustration against the Google Vision
    SafeSearch API and flags if explicit content detected.
    Arguments:
    ---
    sub : ImageSubmission(Object)
    returns:
    ---
    JSON: {"SubmissionID": int, "IsFlagged": boolean, "reason": }
    """
    # fetch file from s3 bucket
    r = get(sub.URL)
    # initalize the sha hashing algorithm
    hash = sha512()
    # update the SHA with the file contents
    hash.update(r.content)
    # assert the the computed hash and the passed hash are the same
    try:
        assert hash.hexdigest() == sub.Checksum
    except AssertionError:
        # return bad hash error with the status_code to an ill-formed request
        return JSONResponse(status_code=422, content={"ERROR": "BAD CHECKSUM"})
    # pass file to the GoogleAPI object to safe search filter the file
    response = await vision.detect_safe_search(r.content)
    # respond with the output dictionary that is returned from
    # GoogleAPI.transcribe() method
    return JSONResponse(status_code=200, content=response)
