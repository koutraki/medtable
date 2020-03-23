import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
  database="MedQuADwithTableNet"
)

def query_DB(article):
  sql_select_Query = "SELECT * FROM MedQuADwithTableNet.tablenet_entities where entity = \'" + article +"\';"
  cursor = mydb.cursor()
  cursor.execute(sql_select_Query)
  records = cursor.fetchall()
  cursor.close()
  return records