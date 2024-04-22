from requests import post

UPLOAD_URL = 'http://127.0.0.1:3000/api/upload'

FILE_PATH = 'uploads/test.txt'

def upload_file():
    with open(FILE_PATH, 'rb') as file:
        files = {'file': file}
        response = post(UPLOAD_URL, files=files)

    print(response.text)

if __name__ == '__main__':
    upload_file()

