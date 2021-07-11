import requests as req
  
import json 
import csv 
import pandas as pd
from pandas.io.json import json_normalize


#isbn = ['1428004858','1443220698','1153587408']
isbn = ['1428004858','1443220698','1153587408','1432500635','0554365464','0974878901','1150050969',
        '1419156314','1409237311','1428025693','1116968525','116146655X','1438574193','116386370X','1421973197','1153725800',
        '147330539x','1771961260','1447404300']
h = {'Authorization': '45647_b8b9533c0c1bc5d09dfce928e4bc787f'}
#resp = req.get("https://api2.isbndb.com/book/1428004858", headers=h)
#print(resp.json())

# now we will open a file for writing 
data_file = open('data_filesss3.csv', 'w') 
  
# create the csv writer object 
csv_writer = csv.writer(data_file) 
  
# Counter variable used for writing  
# headers to the CSV file 
count = 0
killme = []
header = []
df = pd.DataFrame()

for i in isbn:
    resp = req.get("https://api2.isbndb.com/book/"+i, headers=h)
    jsontxt = resp.json()
    book_df = pd.DataFrame(jsontxt['book'])
    #print(book_df)
    df = df.append(book_df, ignore_index=True)
print(df)
data_file.close()

""" 
for book in book_info: 
    if count == 0: 
  
        # Writing headers of CSV file 
        header = book.keys() 
        csv_writer.writerow(header) 
        count += 1
  
    # Writing data of CSV file 
    csv_writer.writerow(book.values()) 


for i in isbn:
    resp = req.get("https://api2.isbndb.com/book/1443220698", headers=h)
    jsontxt = resp.json()
    print(jsontxt)
    for x,y in jsontxt['book'].items():
        z = json.dumps(y)
        header.append(x)
        killme.append(z)
    #print(header)
    #print(killme)
    #if count == 0:
  #      csv_writer.writerow([header]) 
 #       count += 1
 #   csv_writer.writerow([killme])
 #   print(jsontxt)
 #   list(killme)
 #   list(header)
    killme = []
    header = []
data_file.close()
    

    jsontxt=json_normalize(jsontxt)
    #book_info = jsontxt['book'] 
    #print("this is flat",flat)

    book_parsed = json.loads(jsontxt)
    book_data = book_parsed['book']

    for books in booked_data: 
        print(books)
        if count == 0:
            header = books.keys() 
            csv_writer.writerow([header]) 
            count += 1
  
    # Writing data of CSV file 
        csv_writer.writerow([books.values()])



    #print(jsontxt)
data_file.close() 
"""
""" 

MyFILE=open("Publishertest.csv","w")


WriteThis="Title,Publisher,Binding\n"                                                             
MyFILE.write(WriteThis)
MyFILE.close()

MyFILE=open("Publishertest.csv", "a")

for items in jsontxt["book"]:
    print(items)             
    Publisher=items[0]
   # Title=items[4]
   # Binding=items[8]
    WriteThis=Publisher+ "," + Title + "," + Binding + "," + "\n"
    MyFILE.write(WriteThis)

MyFILE.close()
""" 
