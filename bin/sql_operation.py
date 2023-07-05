import MySQLdb
import datetime
import pandas as pd
import csv
import os
import obtain_data #Local file

try:
    #Connect to SQL database.
    db = MySQLdb.connect("localhost","user","password","findata")
except:
    print('Reminder: Please connect to SQL database first before running this program!')
    quit()

#Used to execute SQL queries.
mycursor = db.cursor()

#Create new SQL database if there is none for the traded financial asset.
def create_sql_table(database, symbol, database_start_date):
    try:
        #See if target database exists.
        d = mycursor.execute('SELECT datetime from %s ORDER BY DATETIME DESC LIMIT 1;' % database)
    except:
        #If not, then, create new database.
        d = mycursor.execute('CREATE TABLE %s (news LONGTEXT, comments LONGTEXT, datetime DATETIME, Open FLOAT, High FLOAT, Low FLOAT, Close FLOAT, Volume FLOAT, Adjusted FLOAT, Dividend FLOAT);' % database)
        update_sql_price(database, symbol, database_start_date) 

#Return latest date of data in SQL database, return format = Python datetime.
def latest_date(database):
    d = mycursor.execute('SELECT datetime from %s ORDER BY DATETIME DESC LIMIT 1;' % database)
    if (d > 0):
        latestDate = mycursor.fetchone()
        return latestDate[0].date()
    else: #d < 0, database is empty.
        return None

#Argument "before" = 1: return nearest date in database before given date.
#Argument "before" = 0: return nearest date in database after given date.
#database: String, date = Python datetime
def nearest_date_before(database, date, before = 1):
    strDate = str(date).split()[0]
    if (before == 1):
        d = mycursor.execute('SELECT datetime from %s where DATETIME < "%s" ORDER BY DATETIME DESC LIMIT 1;' % (database, strDate))
    else:
        d = mycursor.execute('SELECT datetime from %s where DATETIME > "%s" ORDER BY DATETIME ASC LIMIT 1;' % (database, strDate))
    nearestDate = mycursor.fetchone() #Format = tuple
    return nearestDate[0] #Format = Python datetime

#Argument last = 1: Return the trading date (format = Python datetime) that is 'num' days ago from 'date'.
#Argument last = 0: Return the trading date (format = Python datetime) that is 'num' days after 'date'.
def last_n_day(database, date, num, last = 1): #Format: database = String, date = Python datetime or string, num = Int
    if (num == 0):
        return date
    else:
        strDate = str(date).split()[0]
        if (last == 1):
            d = mycursor.execute('SELECT datetime FROM %s where DATETIME < "%s" ORDER BY DATETIME DESC LIMIT %s;' % (database, strDate, num))
        else:
            d = mycursor.execute('SELECT datetime FROM %s where DATETIME > "%s" ORDER BY DATETIME ASC LIMIT %s;' % (database, strDate, num))
        dates = mycursor.fetchall()
        if (len(dates)):
            return dates[len(dates) - 1][0] #Format = Python datetime
        else:
            #There is no relevant dates in the database
            return None

#Extract all numeric price data: ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted'] unless specified.
#Return pandas DataFrame.
def extract_all_numeric_data(database, startDate, endDate, typeOfData = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted']):
    d = mycursor.execute('select datetime, %s from %s where datetime > "%s" and datetime < "%s";' % (', '.join(typeOfData), database, startDate, endDate))
    df = mycursor.fetchall()
    return pd.DataFrame([i[1:len(i)] for i in (item for item in df)], columns = typeOfData, index = [i[0] for i in (item for item in df)])

#take sentiment sample data for training.
def extract_sentiment():
    d = mycursor.execute('select * from Sentiment;')
    #If the database is non-empty.
    if (d > 0):
        df = mycursor.fetchall()
        return pd.DataFrame({'Sentm_Scr': [i[0] for i in df], 'Sentence': [i[1] for i in df]})
    else:
        return pd.DataFrame([]) #Empty dataframe

#Extract news, comments from SQL database.
#database: string, colName: string, input = 'news' or 'comments', date: Python datetime
def extract_text(database, colName, date):
    nxtDate = date + datetime.timedelta(days = 1)
    strDate = str(date).split()[0]
    strNxtDate = str(nxtDate).split()[0]
    d = mycursor.execute('select %s from %s WHERE DATETIME > "%s" AND DATETIME < "%s"' % (colName, database, strDate, strNxtDate))
    if (d > 0): #There is text in SQL database.
        fetchText = mycursor.fetchall()
        if (fetchText[0][0] is not None):
            return fetchText[0][0] #Format = string
        else:
            return ''
    else:
        return ''

#Extract most recent 'numItem' data items. 'numItem' is specified in advance.
def extract_data_num(database, startDate, numItem):
    d = mycursor.execute('select datetime, Close from %s where datetime > "%s" ORDER BY datetime asc LIMIT %s;' % (database, startDate, numItem));
    df = mycursor.fetchall()
    #return a pandas dataframe, index = date, content = price of that date.
    return pd.DataFrame([i[1] for i in df], columns = ['Close'], index = [i[0] for i in df])

#Write text into column 'news' in each table of SQL database.
def sql_text_write(news_text, database, strlatestDate, strNxtlatestDate, columnName):
    d = mycursor.execute('SELECT * from %s WHERE DATETIME > "%s" and datetime < "%s" AND %s IS null;' % (database, strlatestDate, strNxtlatestDate, columnName))
    #news is NULL in database.
    if (d > 0):
        mycursor.execute('update %s SET %s = "%s" WHERE datetime > "%s" and datetime < "%s";' % (database, columnName, news_text, strlatestDate, strNxtlatestDate))
    #news is not NULL in database
    else:
        mycursor.execute('update %s SET %s = concat(%s, "\n----\n%s") WHERE datetime > "%s" and datetime < "%s";' % (database, columnName, columnName, news_text, strlatestDate, strNxtlatestDate))
    db.commit() #make changes to table.

#Return earliest date of data in SQL database, format = Python datetime, date.
def earliest_date(database):
    d = mycursor.execute('SELECT datetime from %s ORDER BY DATETIME ASC LIMIT 1;' % database)
    latestDate = mycursor.fetchone()
    return latestDate[0].date()

#Check if data in SQL is updated.
def info_updateOrNot(database):
    latestDate = latest_date(database)
    if latestDate == None or (latestDate < datetime.datetime.today().date() - datetime.timedelta(days = 1)):
        #Update database with latest stock data.
        return False
    else:
        return True

#Transfer financial data from R to SQL.
#If SQL is empty, then add data starting from 2020 Jan 1.
#If SQL contains some outdated data, then add new data.
#This function calls obtain_data.py, obtain_data.py calls quantmod_r.r
def update_sql_price(database, symbol, db_start_date):
    d = mycursor.execute("select datetime from " + database + " order by datetime desc limit 1;")
    if (d == 0): #table is empty.
        #Starting collecting data from 2020 Jan 1.
        latestDate = str(db_start_date).split()[0] 
    else:
        #return the first row of the result.
        latestDate = mycursor.fetchone() 
        latestDate = latestDate[0] + datetime.timedelta(days = 1)
        latestDate = str(latestDate).split()[0] #Convert to string format
    if (str(datetime.datetime.today()).split()[0] == latestDate):
        print('Database', database, 'is already up-to-date.')
    else:
        df = obtain_data.obtainData(symbol, latestDate) #Obtain financial data
        df['time'] = list(df.index)
        #Use linear interpolation to deal with missing data/NA.
        df = df.interpolate()
        sql = """INSERT INTO """ + database + """ (datetime, Open, High, Low, Close, Volume, Adjusted) VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        val = list(map(lambda x1, x2, x3, x4, x5, x6, x7: tuple([x7, x1, x2, x3, x4, x5, x6]),
                       [i for i in df.iloc[:, 0]], [i for i in df.iloc[:, 1]], 
                       [i for i in df.iloc[:, 2]], [i for i in df.iloc[:, 3]], 
                       [i for i in df.iloc[:, 4]], [i for i in df.iloc[:, 5]],
                       [(i + ' 16:00:00') for i in df.iloc[:, 6]]))
        mycursor.executemany(sql, val)
        db.commit() #make changes to table.
