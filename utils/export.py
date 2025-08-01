# utils/export.py
import zipfile
import io

def create_project_zip(code, assets=None):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("main.py", code)
        zip_file.writestr("README.md", "# Generated by AI Coding Genius Omega")
        if assets:
            for name, data in assets.items():
                zip_file.writestr(f"assets/{name}", data)
    zip_buffer.seek(0)
    return zip_buffer
