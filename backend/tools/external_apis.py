import httpx
import logging
from typing import Dict, Any, List, Optional
from backend.config import (
    EDUCATION_API_KEY,
    COLLEGE_DATA_API_URL,
    SCHOLARSHIP_REGISTRY_URL,
)

logger = logging.getLogger(__name__)


class ExternalAPIService:
    """Service wrapper for communicating with external education, college, and scholarship APIs."""

    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {EDUCATION_API_KEY}" if EDUCATION_API_KEY else "",
        }
        self.college_api_url = COLLEGE_DATA_API_URL or "https://api.educationdata.org/v1"
        self.scholarship_api_url = SCHOLARSHIP_REGISTRY_URL or "https://api.scholarships.gov/v1"

    async def get_college_details(self, college_name: str) -> Dict[str, Any]:
        """Fetch detailed information, accreditation, and branch seat matrix for a college."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    f"{self.college_api_url}/colleges/search",
                    params={"name": college_name},
                    headers=self.headers,
                )
                if response.status_code == 200:
                    return response.json()
                logger.error(f"College API error {response.status_code}: {response.text}")
                return {"error": f"Failed to retrieve data for {college_name}"}
            except httpx.RequestError as exc:
                logger.exception(f"HTTP request failed: {exc}")
                return {"error": "College service is currently unreachable"}

    async def fetch_scholarship_matches(
        self, qualification: str, state: str, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query official registries for active government and private scholarships."""
        params = {"qualification": qualification, "state": state}
        if category:
            params["category"] = category

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    f"{self.scholarship_api_url}/scholarships",
                    params=params,
                    headers=self.headers,
                )
                if response.status_code == 200:
                    return response.json().get("results", [])
                return []
            except httpx.RequestError:
                logger.exception("Failed to connect to scholarship registry API")
                return []

    async def verify_exam_roll_number(
        self, exam_code: str, roll_number: str
    ) -> Dict[str, Any]:
        """Optionally verify student scorecard/marksheet records via official exam board APIs."""
        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                response = await client.post(
                    f"{self.college_api_url}/exams/verify",
                    json={"exam_code": exam_code, "roll_number": roll_number},
                    headers=self.headers,
                )
                if response.status_code == 200:
                    return response.json()
                return {"verified": False, "message": "Record not found"}
            except httpx.RequestError:
                return {"verified": False, "message": "Verification service offline"}


# Instantiated service instance for tool usage
external_api_service = ExternalAPIService()