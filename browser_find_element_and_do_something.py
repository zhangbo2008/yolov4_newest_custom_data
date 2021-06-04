import datetime
import shutil
import os
# import automagica.activities as all_method
import sys
import time
import base64
sys.path.append('..')
sys.path.append('.')
sys.path.append('../')
sys.path.append('..')
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
options = webdriver.ChromeOptions()
chrome_options = options
options.add_experimental_option('excludeSwitches', ['enable-automation'])
dic_for_window ={}
import pickle

from env_variable import *
import binascii
import io

'''
这个代码使用非常复杂所以写一些demo提供参考:
-p  id  \"sb_form_q\"     send_keys(\"abc\")
-p  id  \"sb_form_q\"     get_attribute(\"id\")
-p  id  \"sb_form_q\"     get_property(\"id\")
-p  id  \"sb_form_q\"     text
'''
try:

    import argparse
# 调用demo:   第一个参数:  xpath|class_name|link_text|partial_link_text|name|tag_name|id|css_selector
    # 第二个参数  "aaa"   #表示第一个参数的值  ===========
    # 第三个参数  'send_keys(\"abc\")'|click()|get_attribute("id")
    #  id  \"sb_form_q\"     end_keys(\"abc\")
    parser = argparse.ArgumentParser()
    parser.add_argument( "-p", type=str,nargs='+')
    args = parser.parse_args()
    if not args.p:
        args.p="f"
    # print(args.url)
    # print(11111111)

    # "dic_for_window[{}]={}\n".format("browser.title", "browser.current_window_handle"),
    # browser='''
    # aaaaaaaaaaaaaaaio.BytesIO

    def create_driver():
        driver = webdriver.Chrome(options=chrome_options)
        with open(session_file, 'wb') as f:
            params = {"session_id": driver.session_id, "server_url": driver.command_executor._url}
            pickle.dump(params, f)
        driver.get(args.url)
        return driver

#==============================下面是chrome每一个的逻辑部分.
    if not os.path.exists(session_file):
        raise 1
    else:
        with open(session_file, 'rb') as f:
            params = pickle.load(f)
            try:
                options = webdriver.ChromeOptions()
                options.add_argument("headless")
                driver = webdriver.Remote(command_executor=params["server_url"],options=options)
                driver.quit()  # 退出start_session新开的空白浏览器
                driver.session_id = params["session_id"]
                # driver.execute_script('window.open("");')
                # driver.get(args.url)

#================处理代码写这个地方.
                act=args.p
                aaa='driver.find_element_by_'+act[0]+"("+act[1]+")"+'.'+act[2]
                data=eval('driver.find_element_by_'+act[0]+"("+act[1]+")"+'.'+act[2])
                # driver.switch_to.active_element.send_keys(args.p)
                # driver.close()
                # driver.switch_to.window(driver.window_handles[-1])
            except:
                raise
    if data:
        import json
        a=json.dumps({'code':0,"msg":data},ensure_ascii=False)
    else:
        import json
        a=json.dumps({'code':0,"msg":'success'},ensure_ascii=False)

    print(a)





except:
    print("{'code':'1','msg':'fault'}")

