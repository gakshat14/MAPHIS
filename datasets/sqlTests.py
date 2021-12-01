import sqlite3
from sqlite3 import Error
import mysql.connector
from pyprojroot import here
from pathlib import Path


def createConnection(host_name:str, user_name:str, user_password:str, db_name:str):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name,
            port= 1433
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

def createDatabase(connection:sqlite3.Connection, query:str):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


connection = createConnection("localhost", "root", '1qaz"WSX3edc', r'UOB/hx21262')
createDatasetQuery = "CREATE DATABASE maphis"
createDatabase(connection, createDatasetQuery)
