from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "AI Procurement CRM API Running"
    }


@router.get("/dashboard")
def dashboard():

    return {

        "companies": 482,

        "suppliers": 176,

        "contacts": 2381,

        "coverage": 94

    }