import io
from bs4 import BeautifulSoup

with io.open('../datasets/shahnameh-ferdosi.htm', 'r', encoding='utf-8') as file:
    html = file.read()


def filter_poems(tag):
    return tag.name == 'span' and tag.has_attr('class') and 'content_text' in tag.get('class') and '****' in tag.get_text()


def filter_labels(tag):
    return tag.name == 'h2' and tag.has_attr('class') and 'content_h2' in tag.get('class')


def filter_poems_labels(tag):
    return filter_poems(tag) or filter_labels(tag)


soup = BeautifulSoup(html, 'html.parser')
poems_and_labels = soup.find_all(filter_poems_labels)

with io.open('../datasets/shahnameh-labeled.csv', 'w', encoding='utf-8') as file:
    file.write('mesra1,mesra2,label\n')
    label = ''
    for item in poems_and_labels:
        if filter_labels(item):
            label = item.get_text()
        elif filter_poems(item) and label.startswith('داستان'):
            mesras = item.get_text().split('****')
            file.write(f'{mesras[0]},{mesras[1]},{label}\n')
