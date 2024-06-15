### INPUTS
INPUT_BROWSER_PATH = "C:/chromedriver.exe"
OUTPUT_FOLDER = "output_files" # where to save files
INPUT_PLAYERS_LIST = [{"url":"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p=NovakDjokovic&f=ACareerqq", "filename":"NovakDjokovic.csv", "playername":"Novak Djokovic"},
                      {"url":"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p=RafaelNadal&f=ACareerqq", "filename":"RafaelNadal.csv", "playername":"Rafael Nadal"},
                      {"url":"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p=RogerFederer&f=ACareerqq", "filename":"RogerFederer.csv", "playername":"Roger Federer"}]
### END OF INPUTS


# install: selenium, lxml
from selenium import webdriver  #install selenium
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
import time
import pprint
import os
import sys
from datetime import datetime
from lxml import html # pip install lxml
from urllib.parse import urljoin
import sqlite3
import csv


class TennisabstractScraper:
    def __init__(self, input_browser_path, input_outfolder, input_playerslist, mode='scrape'):
        self.TIME_PAUSE = 1.0 # pause for xpath, nothing to do with input pause!
        
        ## set inputs
        self.PATH_TO_BROWSER = input_browser_path
        self.output_folder = input_outfolder
        self.players_to_scrape = input_playerslist

        ## start chrome
        if mode == 'scrape':
            self.driver = self.start_driver_normal()
    
        return


    def scrape_players(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        
        print("Scraping...")
        # iterate across each player and save their data to a file
        for player_index, player_to_scrape in enumerate(self.players_to_scrape):
            try:
                # open the link
                self.destroy_handles_and_create_new_one()
                self.driver.get(player_to_scrape["url"])
                wait_for_data = self.wait_by_xpath("//table[@id='maintable']//table[@id='matches']", 30)
                if wait_for_data == 0:
                    print("Couldn't find data at", player_to_scrape["url"])
                    continue
                
                # parse out data into a list
                innerHTML = self.driver.execute_script("return document.body.innerHTML")
                tree = html.document_fromstring(innerHTML)
                parsed_data = self.parse_data_from_html(tree, player_to_scrape["playername"])

                # write it into output file
                #print("Found", len(parsed_data), "matches for", player_to_scrape["playername"])
                self.write_data(parsed_data, os.path.join(self.output_folder, player_to_scrape["filename"]))
                #pprint.pprint(parsed_data[0])
                #print([key for key in parsed_data[0]])
                
            except KeyboardInterrupt:
                print("Manual interrupt, stop!")
                break
            except Exception as exc:
                print("An exception while scraping for", player_to_scrape["url"], ":", repr(exc))
                continue

        # close chrome
        self.driver.quit()
        
        return


    def parse_data_from_html(self, input_tree, observed_player_name):
        data_to_return = []

        # get matches table
        match_table_el = input_tree.xpath("//table[@id='maintable']//table[@id='matches']")
        if len(match_table_el) != 0:
            # get match headers, empty header should be meta
            match_headers = [header_el.text_content().strip() for header_el in match_table_el[0].xpath("./thead/tr[1]/th")]
            if "" in match_headers:
                if match_headers.count("") == 1:
                    match_headers[match_headers.index("")] = 'meta_info'

            # go over matches and add info
            match_row_els = match_table_el[0].xpath("./tbody/tr/td/..")
            for match_row_el in match_row_els:
                datacolumn_els = match_row_el.xpath("./td")
                if len(datacolumn_els) == len(match_headers) and len(match_headers) != 0:
                    this_match = {match_headers[header_index]:datacolumn_els[header_index].text_content().replace("\xa0", " ").strip() for header_index in range(0, len(match_headers))}

                    # figure out opponent's name and who whon
                    this_match["Observed Player Name"] = observed_player_name
                    this_match["Opponent Player Name"] = None
                    this_match["Observed Player W/L"] = None

                    if "meta_info" in match_headers:
                        opponent_name_el = datacolumn_els[match_headers.index("meta_info")].xpath("./span/a[contains(@href, '?p=')]")
                        if len(opponent_name_el) != 0:
                            this_match["Opponent Player Name"] = opponent_name_el[0].text_content().strip()

                        # figure out w/l
                        observed_before_like_sign = datacolumn_els[match_headers.index("meta_info")].xpath("./span[@class='likelink h2hclick']/preceding-sibling::span[contains(@style, 'font-weight: bold')]")
                        observed_after_like_sign = datacolumn_els[match_headers.index("meta_info")].xpath("./span[@class='likelink h2hclick']/following-sibling::span[contains(@style, 'font-weight: bold')]")
                        if len(observed_before_like_sign) != 0 and len(observed_after_like_sign) == 0:
                            this_match["Observed Player W/L"] = 'W'
                        elif len(observed_before_like_sign) == 0 and len(observed_after_like_sign) != 0:
                            this_match["Observed Player W/L"] = 'L'
                        else:
                            print("Couldn't figure out W/L for one match!")
                            pass
                        
                    # add this match
                    data_to_return.append(this_match)
                    
                else:
                    print("Can't add one match row since the number of columns isn't right!")
            
        return data_to_return



    def write_data(self, input_list, input_filepath):
        headers = ['Date', 'Tournament', 'Surface', 'Rd', 'Rk', 'vRk', 'meta_info', 'Score', 'Observed Player Name', 'Opponent Player Name', 'Observed Player W/L',
                   'More', 'DR', 'A%', 'DF%', '1stIn', '1st%', '2nd%', 'BPSvd', 'Time']
        outfile = open(input_filepath, 'w', newline='', encoding='utf-8')
        writer = csv.writer(outfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        ccc = writer.writerow(headers)

        # write each match
        for match_to_write in input_list:
            row_to_write = []
            for header_item in headers:
                if header_item in match_to_write:
                    row_to_write.append(match_to_write[header_item])
                else:
                    row_to_write.append(None)

            ccc = writer.writerow(row_to_write)

        outfile.close()
        print("Created output file with", len(input_list), "matches at:", input_filepath)
        return
    
    

    def wait_by_xpath(self, xp, how_long_to_wait): # xp is string, how_long_to_wait float - the number of seconds to wait
        try:
            WebDriverWait(self.driver, how_long_to_wait).until(EC.presence_of_element_located((By.XPATH, xp)) )
            time.sleep(self.TIME_PAUSE)
            return 1 # success
        except TimeoutException:
            print ("Too much time has passed while waiting for", xp)
            return 0 # fail



    def start_driver_normal(self):
        serv = Service(self.PATH_TO_BROWSER)
        normal_driver = webdriver.Chrome(service=serv)
        normal_driver.maximize_window()
        return normal_driver


    def start_driver_headless(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("window-size=1920,1080")
        headless_driver = webdriver.Chrome(self.PATH_TO_BROWSER, options=chrome_options)
        return headless_driver


    def destroy_handles_and_create_new_one(self):
        # call it before opening an url
        while 1:
            initial_handles = self.driver.window_handles
            self.driver.execute_script("window.open();")
            handles_after_opening = self.driver.window_handles
            if len(handles_after_opening) > len(initial_handles):
                break
            else:
                print("Couldn't open a handle!")
                time.sleep(10.0)
                continue
            
        added_handle = []
        for handle in handles_after_opening:
            if handle in initial_handles:
                self.driver.switch_to.window(handle)
                self.driver.close()
            else:
                added_handle.append(handle)

        self.driver.switch_to.window(added_handle[0])
        return

    
if __name__ == '__main__':
    scraper_instance = TennisabstractScraper(INPUT_BROWSER_PATH, OUTPUT_FOLDER, INPUT_PLAYERS_LIST)
    scraper_instance.scrape_players()
