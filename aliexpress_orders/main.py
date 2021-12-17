from selenium import webdriver
from time import sleep
import pandas as pd
from tqdm import trange, tqdm

driver = webdriver.Chrome(r'/Users/aliaksandr.lashkov/Applications/chromedriver')
driver.get('https://login.aliexpress.com')
# driver.switch_to_frame(0)
username = driver.find_element_by_name("fm-login-id")
username.send_keys('pop4irka@gmail.com')
sleep(1)
password = driver.find_element_by_name("fm-login-password")
password.send_keys('Dtctkfz45pyflyj')
sleep(1)
submit = driver.find_element_by_css_selector('button.fm-button')
submit.click()

orders_page = 'https://trade.aliexpress.com/orderList.htm'
driver.get(orders_page)
sleep(5)

paginator = driver.find_element_by_class_name('ui-pagination-navi.util-left')
max_page = max(list(map(int, paginator.text.split('\n')[1:-1])))
df = pd.DataFrame(columns=['img', 'desc', 'tracking_number', 'status', 'tracking_steps'])
for page_num in trange(1, max_page + 1, desc=f'working on page'):
    driver.get(orders_page)
    sleep(2)
    if page_num != 1:
        page_btn = driver.find_element_by_xpath(f"//a[@class='ui-goto-page' and text()='{page_num}']")
        page_btn.click()
#         sleep(5)

    page_orders = driver.find_elements_by_class_name('order-item-wraper')
    order_ids = []
    for order in tqdm(page_orders, desc=f'working on page orders'):
        order_id = order.find_element_by_class_name('info-body').text
        order_ids.append(order_id)
        img_url = order.find_element_by_tag_name('img').get_attribute('src')
        df.loc[order_id, 'img'] = img_url
        df.loc[order_id, 'desc'] = order.find_element_by_class_name('baobei-name').get_attribute('title')
        df.loc[order_id, 'status'] = order.find_element_by_class_name('f-left').text

    for order_id in order_ids:
        driver.get(f'https://track.aliexpress.com/logisticsdetail.htm?tradeId={order_id}')
        sleep(1)
        tracking_numbers = driver.find_elements_by_class_name('tracking-no')
        if tracking_numbers:
            df.loc[order_id, 'tracking_number'] = tracking_numbers[0].text
        ship_steps = driver.find_elements_by_css_selector('.step')
        if ship_steps:
            tracking_steps = []
            for step in ship_steps:
                time = step.find_element_by_css_selector('.step-time').text.replace('\n', ' ')
                step_content = step.find_element_by_css_selector('.step-content').text.replace('\n', ' ')
                tracking_steps.append(f'{time} {step_content}')
            df.loc[order_id, 'tracking_steps'] = '\n'.join(tracking_steps)





# import pandas as pd
# from IPython.core.display import display,HTML
#
# df = pd.DataFrame([['A231', 'Book', 5, 3, 150],
#                    ['M441', 'Magic Staff', 10, 7, 200]],
#                    columns = ['Code', 'Name', 'Price', 'Net', 'Sales'])
#
# # your images
# images1 = ['https://vignette.wikia.nocookie.net/2007scape/images/7/7a/Mage%27s_book_detail.png/revision/latest?cb=20180310083825',
#           'https://i.pinimg.com/originals/d9/5c/9b/d95c9ba809aa9dd4cb519a225af40f2b.png']
#
#
# images2 = ['https://static3.srcdn.com/wordpress/wp-content/uploads/2020/07/Quidditch.jpg?q=50&fit=crop&w=960&h=500&dpr=1.5',
#            'https://specials-images.forbesimg.com/imageserve/5e160edc9318b800069388e8/960x0.jpg?fit=scale']
#
# df['imageUrls'] = images1
# df['otherImageUrls'] = images2
#
#
# # convert your links to html tags
# def path_to_image_html(path):
#     return '<img src="'+ path + '" width="60" >'
#
# pd.set_option('display.max_colwidth', None)
#
# image_cols = ['imageUrls', 'otherImageUrls']  #<- define which columns will be used to convert to html
#
# # Create the dictionariy to be passed as formatters
# format_dict = {}
# for image_col in image_cols:
#     format_dict[image_col] = path_to_image_html
#
#
# display(HTML(df.to_html(escape=False ,formatters=format_dict)))