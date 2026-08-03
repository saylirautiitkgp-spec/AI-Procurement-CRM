const API_URL = "http://127.0.0.1:8000";

export async function getCompanies() {
    const response = await fetch(`${API_URL}/companies`);
    return response.json();
}

export async function getWorkflow() {
    const response = await fetch(`${API_URL}/workflow`);
    return response.json();
}

export async function getDashboard() {
    const response = await fetch(`${API_URL}/dashboard`);
    return response.json();
}