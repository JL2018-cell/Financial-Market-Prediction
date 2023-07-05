#Store web-scrapped news in sql instead of local txt files.
from typing import Text
from selenium import webdriver
from selenium.webdriver.common import by
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import selenium.webdriver.support.expected_conditions as EC
from getpass import getpass
import time
import os
import datetime
import csv
import sql_operation

#Web-scrap news
#format: String, String, Python datetime
def scrap_news(database, keyword, region, latestDate):

    USER_ID = input('HKU Portal UID / Library card number: ')
    PASSWORD = getpass('PIN: ')

    #Used to remove those characters in the name of news.
    FORBIDDEN_CHARS = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    
    options = webdriver.ChromeOptions()
    #Language of browser = English language
    options.add_argument("--lang=en");

    try:
        #Launch browser
        driver = webdriver.Chrome(
            executable_path='./chromedriver.exe',
            chrome_options=options
        )
    except:
        print('Please put a chromedriver for web-scrapping!')
        quit()

    #Browse the link.
    FACTIVA_URL = 'https://julac-hku.alma.exlibrisgroup.com/view/action/uresolver.do?operation=resolveService&package_service_id=15754959500003414&institutionId=3414&customerId=3405'
    driver.get(FACTIVA_URL) 

    #Input user name.
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="userid"]')))
    userIdField = driver.find_element_by_css_selector('input[name="userid"]')
    userIdField.send_keys(USER_ID)

    #Input password.
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="password"]')))
    passwordField = driver.find_element_by_css_selector('input[name="password"]')
    passwordField.send_keys(PASSWORD)

    #Click login button.
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[name="submit"]')))
    submitButton = driver.find_element_by_css_selector('input[name="submit"]')
    submitButton.click()

    #Click textbox to start searching.
    WebDriverWait(driver, 600).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[title="Search"]')))
    searchButton = driver.find_element_by_css_selector('a[title="Search"]')
    searchButton.click() 

    #Need to wait!
    #Adjust Factiva language to English.
    for i in range(1000):
        try:
            settingButton = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'span[class="fi-two fi_settings"]')))
            settingButton.click()
            LanguageButton = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[class="language"]')))
            LanguageButton.click()
            EnglishButton = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[title="English"]')))
            EnglishButton.click()
            WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'span[class="fi-two fi_settings"]')))
            break
        except:
            if (i < 1000 - 1):
                continue
            else:
                print("Sorry, web-scrapping fails!")
                quit()

    #Select date of news to be web-scrapped.
    dateSelector = WebDriverWait(driver, 100).until(EC.element_to_be_clickable((By.ID, 'dr')))
    dateSelector.click()
    wkAgoDate = latestDate - datetime.timedelta(days = 7) #1 week
    enY = str(latestDate.year) #end year
    enM = str(latestDate.month) #end month
    enD = str(latestDate.day) #end day
    stY = str(wkAgoDate.year) #start year
    stM = str(wkAgoDate.month) #start month
    stD = str(wkAgoDate.day) #start day
    #Change the time interval to be searched.
    allDatesOption = driver.find_element_by_css_selector('option[value="Custom"]')
    allDatesOption.click()
    startDay = driver.find_element_by_css_selector('input[name="frd"]')
    startDay.send_keys(stD) #Day
    startMth = driver.find_element_by_css_selector('input[name="frm"]')
    startMth.send_keys(stM) #Month
    startYr = driver.find_element_by_css_selector('input[name="fry"]')
    startYr.send_keys(stY) #Year
    endDay = driver.find_element_by_css_selector('input[name="tod"]')
    endDay.send_keys(enD) #Day
    endMth = driver.find_element_by_css_selector('input[name="tom"]')
    endMth.send_keys(enM) #Month
    endYr = driver.find_element_by_css_selector('input[name="toy"]')
    endYr.send_keys(enY) #Year
    dateSelector.click()
    
    #WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'textarea[class="ace_text-input"]')))
    txtField = driver.find_element_by_css_selector('textarea[class="ace_text-input"]')
    txtField.send_keys(keyword)

    #Fill in region field if region is not an empty string.
    if (len(region)): 
        regionPicker = driver.find_element_by_css_selector('#reTab > .pnlTabArrow')
        regionPicker.click()
        regionField = driver.find_element_by_id('reTxt')
        regionField.send_keys(region)
        #Search region
        regionFieldSubmitButton = driver.find_element_by_id('reLkp')
        regionFieldSubmitButton.click()
        if (region == 'Hong Kong'):
            hongKongOption = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[code="re_hkong"]')))
            hongKongOption.click()
        elif (region == 'United States'):
            usaOption = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[code="re_usa"]')))
            usaOption.click()
        elif (region == 'China'):
            chinaOption = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[code="re_china"]')))
            chinaOption.click()
        else:
            firstOption = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[code^="re_"]')))
            firstOption.click()

    #Start searching.
    searchSubmitButton = driver.find_element_by_id('btnSearchBottom')
    searchSubmitButton.click()

    #Record time when this program startsweb-scrapping
    WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'sources')))
    start_Scrap_Time = datetime.datetime.now()

    #Sometimes there are too many news to be 
    #So, count_page_limit is imposed to limit web-scrapping first few pages.
    count_page_limit = 2
    count_page = 0
    while(count_page < count_page_limit):
        #Prevent irrelevant web elements masking target button.
        try:
            overlayBackground = driver.find_element_by_id('__overlayBackground')
            while ('display: none' not in overlayBackground.get_attribute('style')):
                pass
        except:
            WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'newsSubjects')))
        WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'newsSubjects')))

        #Obtain news headline.
        headlinesTable = WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'headlines')))
        allHeadlines = driver.find_elements_by_class_name('enHeadline')
        #Find format of news in a webpage: txt? pdf?
        allFormats = driver.find_elements_by_css_selector('img[title^="Factiva Licensed Content"]')

        #Write news content
        for hdNum, headline in enumerate(allHeadlines):
            try:
                #Obtain headline of news.
                filename = headline.text
                #Remove forbidden characters in news headline.
                for char in FORBIDDEN_CHARS:
                    if char in filename:
                        filename =  filename.replace(char, '_')
                try:
                    #Ignore news in PDF format.
                    src = allFormats[hdNum].get_attribute('src')
                    if ('PDF' in src):
                        print('Src =', src)
                        print('Ignore PDF file.')
                        continue
                except:
                    continue
                #Browse content of news.
                headline.click()
                print('Length of allHeadlines =', len(allHeadlines))
                print('Writing news to database:', database)
                try:
                    #See of content of news is loaded successfully.
                    WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.XPATH, '//*[starts-with(@id, "aRelIndex_")]')))
                except:
                    #If no, then refresh browser and reload content of news.
                    driver.refresh()
                    headlinesTable = WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'headlines')))
                    try:
                        WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, '__overlayBackground')))
                        overlayBackground = driver.find_element_by_id('__overlayBackground')
                        while ('display: none' not in overlayBackground.get_attribute('style')):
                            pass
                    except:
                        WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.ID, 'headlines')))
                    allHeadlines = driver.find_elements_by_class_name('zhtwHeadline')
                    headline.click()
                #Obtain content of news.
                articleBody = WebDriverWait(driver, 6000).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'div[class="article enArticle"]')))
                #Calculate date of news.
                strlatestDate = str(latestDate).split()[0] #Comvert form datetime format to string format
                strnxtlatestDate = str(latestDate + datetime.timedelta(days = 1)).split()[0] #Convert form datetime format to string format
                #Write news to SQL database.
                sql_operation.sql_text_write(articleBody.text, database, strlatestDate, strnxtlatestDate, 'news')
                newFilename = articleBody.text[articleBody.text.rfind('Document '):].strip()
            except:
                #The news is duplicated. Skip web-scrapping this news.
                print('The headline is a duplicate.')
            try:
                #Return no matter the healine is successfully clicked or not.
                returnLink = driver.find_element_by_id('returnToHeadlines')
                returnLink.click()
            except:
                print("Return to news headline page.")
        try:
            print("Try continuing scrapping in next page.")
            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'nextItem')))
            nextPage = driver.find_element(By.CLASS_NAME, 'nextItem')
            if (nextPage.is_enabled() and nextPage.is_displayed() and count_page < count_page_limit):
                count_page += 1
                nextPage.click()
                print("Continue scrapping in next page.")
        except:
            #Finish web-scrapping.
            print("Scrap news to", database, "ends.")
            break
    
    end_Scrap_Time = datetime.datetime.now()
    print("Time taken for scraping =", end_Scrap_Time - start_Scrap_Time)
    print("Web-scrapper of", database, "ends now.")
    #Close bowser.
    driver.quit()
