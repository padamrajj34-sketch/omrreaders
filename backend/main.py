from __future__ import annotations
import json, os, re, uuid, secrets
from pathlib import Path
from typing import Annotated
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DATA_DIR = Path(os.getenv("OMR_DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS = "ABCD"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
app = FastAPI(title="ClearMark OMR", version="1.0.0", description="OMR test setup and grading API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

security = HTTPBasic(auto_error=False)
TEACHER_USERNAME = os.getenv("TEACHER_USERNAME", "")
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "")

def require_teacher(credentials: Annotated[HTTPBasicCredentials | None, Depends(security)]) -> str:
    if not TEACHER_USERNAME or not TEACHER_PASSWORD:
        raise HTTPException(503, "Teacher access is not configured. Set TEACHER_USERNAME and TEACHER_PASSWORD.")
    if credentials is None or not secrets.compare_digest(credentials.username, TEACHER_USERNAME) or not secrets.compare_digest(credentials.password, TEACHER_PASSWORD):
        raise HTTPException(401, "Teacher sign-in failed", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

class TestSetup(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    question_count: int = Field(ge=1, le=100)
    answer_key: list[str]

def test_path(test_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", test_id): raise HTTPException(400, "Invalid test ID")
    return DATA_DIR / f"{test_id}.json"

def read_test(test_id: str) -> dict:
    path = test_path(test_id)
    if not path.exists(): raise HTTPException(404, "Test not found")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/health")
def health(): return {"status": "ok", "scanner": "kathmandu-100-v1"}

@app.get("/api/tests")
def list_tests():
    result=[]
    for path in sorted(DATA_DIR.glob("*.json"), key=lambda p:p.stat().st_mtime, reverse=True):
        try:
            t=json.loads(path.read_text(encoding="utf-8")); result.append({"id":path.stem,"title":t["title"],"question_count":t["question_count"]})
        except (OSError,ValueError,KeyError): continue
    return result

@app.get("/api/teacher/verify")
def teacher_verify(username: Annotated[str, Depends(require_teacher)]):
    return {"role": "teacher_admin", "username": username}

@app.post("/api/tests")
def create_test(payload: TestSetup, username: Annotated[str, Depends(require_teacher)]):
    if len(payload.answer_key) != payload.question_count: raise HTTPException(422,"Answer key length must match question count")
    if any(a not in OPTIONS for a in payload.answer_key): raise HTTPException(422,"Answers must be A, B, C, or D")
    test_id=uuid.uuid4().hex[:12]; record={"id":test_id,"created_by":username,**payload.model_dump()}
    test_path(test_id).write_text(json.dumps(record,indent=2),encoding="utf-8")
    return record

@app.get("/api/tests/{test_id}")
def get_test(test_id: str, username: Annotated[str, Depends(require_teacher)]): return read_test(test_id)

def order_points(points):
    ordered=np.zeros((4,2),dtype=np.float32); sums=points.sum(axis=1); diffs=np.diff(points,axis=1).ravel()
    ordered[0],ordered[2]=points[np.argmin(sums)],points[np.argmax(sums)]
    ordered[1],ordered[3]=points[np.argmin(diffs)],points[np.argmax(diffs)]
    return ordered

def rectify_sheet(image):
    """Correct perspective from the paper outline; retain the input frame if corners are hidden."""
    h,w=image.shape[:2]; scale=min(1.0,1800/max(h,w))
    search=cv2.resize(image,(int(w*scale),int(h*scale))) if scale<1 else image
    gray=cv2.cvtColor(search,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),45,140)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE); corners=None
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:20]:
        polygon=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(polygon)==4 and cv2.contourArea(polygon)>search.shape[0]*search.shape[1]*.35:
            corners=polygon.reshape(4,2).astype(np.float32)/scale; break
    if corners is None: return image
    src=order_points(corners); width,height=1000,1414
    dst=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    return cv2.warpPerspective(image,cv2.getPerspectiveTransform(src,dst),(width,height))

def bubble_centers(question_count,width,height):
    """Bubble locations calibrated to the supplied 828x1171 rendering of the A4 reference."""
    centers=[]
    for q in range(question_count):
        column,row=divmod(q,20); y=(476+row*(616/19))/1171*height; left=[59,205,349,495,639][column]
        centers.append([(int((left+offset)/828*width),int(y)) for offset in (32,64,97,129)])
    return centers

def detect_answers(image,question_count):
    sheet=rectify_sheet(image); gray=cv2.GaussianBlur(cv2.cvtColor(sheet,cv2.COLOR_BGR2GRAY),(3,3),0)
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,17)
    answers=[]; confidence=[]; height,width=gray.shape; radius=max(4,int(width*.0105))
    for centers in bubble_centers(question_count,width,height):
        density=[]
        for x,y in centers:
            x0,x1=max(0,x-radius),min(width,x+radius+1); y0,y1=max(0,y-radius),min(height,y+radius+1); patch=binary[y0:y1,x0:x1]
            yy,xx=np.ogrid[:patch.shape[0],:patch.shape[1]]; cy,cx=y-y0,x-x0
            mask=(xx-cx)**2+(yy-cy)**2 <= (radius*.62)**2
            density.append(float(np.count_nonzero(patch[mask]))/max(1,int(mask.sum())))
        order=np.argsort(density)[::-1]; best,runner=float(density[order[0]]),float(density[order[1]])
        marked=[int(i) for i,value in enumerate(density) if value>=.20]
        multiple=len(marked)>1
        selected=int(order[0]) if not multiple and best>=.20 and best-runner>=.045 else None
        answers.append(OPTIONS[selected] if selected is not None else None)
        status="multiple_marks" if multiple else ("detected" if selected is not None else "blank_or_ambiguous")
        confidence.append({"mark_density":round(best,3),"margin":round(best-runner,3),"marked_options":[OPTIONS[i] for i in marked],"status":status})
    return answers,confidence

@app.post("/api/grade")
async def grade_omr(image: Annotated[UploadFile,File(...)],test_id: Annotated[str,Form(...)]):
    record=read_test(test_id)
    if image.content_type not in {"image/jpeg","image/png","image/webp"}: raise HTTPException(415,"Upload a JPG, PNG, or WEBP image")
    raw=await image.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(413,"Image must be 12 MB or smaller")
    if not raw: raise HTTPException(400,"Uploaded image is empty")
    decoded=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if decoded is None: raise HTTPException(400,"Could not read image. Try a clear JPG or PNG photo.")
    if min(decoded.shape[:2])<700: raise HTTPException(422,"Image is too small. Use a clear scan or photo at least 700 pixels across.")
    answers,confidence=detect_answers(decoded,record["question_count"]); key=record["answer_key"]
    correct=sum(a==k for a,k in zip(answers,key)); unanswered=sum(a is None for a in answers); incorrect=record["question_count"]-correct-unanswered
    return {"test_id":test_id,"test_title":record["title"],"question_count":record["question_count"],"score":correct,"percentage":round(correct/record["question_count"]*100,1),"correct":correct,"incorrect":incorrect,"unanswered":unanswered,"student_answers":answers,"answer_key":key,"confidence":confidence}

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
