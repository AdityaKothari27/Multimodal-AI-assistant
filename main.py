import kdbai_client as kdbai
import os
from dotenv import load_dotenv

load_dotenv()

kdbapikey = os.getenv("KDB_API_KEY")
session = kdbai.Session(endpoint="https://cloud.kdb.ai/instance/1hp406mlym", api_key="kdbapikey")


db = session.database('default')