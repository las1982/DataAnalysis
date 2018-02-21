from googletrans import Translator
import csv
from profitero_data_scientist.Enums import *
import pandas as pd


class PrepareData:
    jd_reviews_data = None
    reviews_data = None
    product_data = None

    def __init__(self):
        # self.make_data()
        pass

    def make_data(self):
        self.make_jd_reviews_data()
        self.make_reviews_data()
        self.make_product_data()
        self.write_data_to_csv_files(self.reviews_data, File.review)
        self.write_data_to_csv_files(self.product_data, File.product)

    def make_jd_reviews_data(self):
        self.jd_reviews_data = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str')

    def make_reviews_data(self):
        self.reviews_data = pd.DataFrame(self.jd_reviews_data[[Field.review]])
        self.reviews_data.dropna(inplace=True)
        self.reviews_data.drop_duplicates(inplace=True)

    def make_product_data(self):
        self.product_data = pd.DataFrame(self.jd_reviews_data[[Field.product_id, Field.product_name]])
        self.product_data.dropna(inplace=True)
        self.product_data.drop_duplicates(inplace=True)

    def fill_review(self):
        pass

    def fill_product(self):
        pass

    def write_data_to_csv_files(self, data, file):
        data.to_csv(file, encoding='utf-8', index=True, quotechar='"', sep=',')

    def translate_review(self):
        pass

    def translate_product(self):
        pass

    def fff(self):
        translator = Translator()
        # print(translator.translate('안녕하세요.'))
        # print(translator.translate('안녕하세요.', dest='ja'))
        # print(translator.translate('veritas lux mea', src='la'))
        # print(translator.translate('收到一打开包装袋就是坏的&hellip;&hellip;找客服，客服让我找京东自营店客服，好吧 找了一看要么退货要么换货，太麻烦了也来不及了，原本想送给表妹的结婚礼物&hellip;&hellip;只好留着自用了 坑', dest='en'))
        texts = [
            '很好吃，口感真心不错',
            '包装很好，没有坏的',
            '膻味太重了！',
            '干果有些受潮，不太好吃，如何能售后维权呢？',
            '可以的，就是盒子小了点',
            '孙子喜欢吃',
            '不怎么样',
            '运行速度很快 外观也很漂亮 配置比ov 强太多',
            '找到儿时的味道',
            '还好吧，家乡的味道',
            '分量挺足，拿回家搬上楼累*我了，包装保持不错，(ฅ&forall;&lt;`๑)╭',
            '好吃。每个味各有各的口感。递哥服务五个星！',
            '挺好，只是感觉不错比普通的好在哪，价贵',
            '每次都买它，狗狗爱吃，但就是气味太大，配方改了？',
            '东西还可以，服务态度太差☠☠',
            '再不会在京东买东西',
            '收到一打开包装袋就是坏的&hellip;&hellip;找客服，客服让我找京东自营店客服，好吧 找了一看要么退货要么换货，太麻烦了也来不及了，原本想送给表妹的结婚礼物&hellip;&hellip;只好留着自用了 坑',
            '灰常好',
            '商品还可以',
            '好评！很不错！非常喜欢！点赞哦！',
            '给女朋友买的，顺丰第二天就到了。她收到后挺惊喜的，包装很扎实，每一个樱桃颗粒都很均匀，没有坏的，她非常喜欢。满意！',
            '怪难吃，还是北京特产？？',
            '不错',
            '从前在恩施的确觉得土豆是特产，比别处好吃。没想到网购的有一半是比较大的普通白心土豆，另一半是较小的黄色的，所以感觉是混装。黄土豆味道还行。',
            '吃多齁人',
            '京东购物感觉很给力，快的速度快，京东产品物美价廉。',
            '买回来家婆说不好，很多坏的',
            '非常适合我家的小狗',
            '婚礼用的，挺好的',
            '不错☜',
            '好吃的，下次买多点',
            '佐料味道不错，肉质就一般了，冰冻过的毕竟还是差点',
            '还不错的饼干宝宝喜欢吃。',
            '还没有吃过，等吃过之后再来追评！希望好吧！！！',
            '给狗狗屯的，暑期交给爸爸帮忙养，吃狗粮方便干净',
            '分量太少太少',
            '老家的味道就是好吃',
            '口味可以 本以为度数很大的 喝起来感觉不错 总体还好',
            '一打开外袋飘出一股淡淡的花香，再打开独立包装香味就浓了些，吃起来也有浓浓的玫瑰味儿，第一次买鲜花饼，感觉还不错',
            '一如既往的快速发货，一如既往的美味，一如既往的支持。',
            '一如既往地，发货快，商品好，赞的。喵喵很爱吃。',
            '京东物流就是快，产品质量不错，无异味，较柔和，拉伸也不错，夏天运动好搭档。',
            '物流实在太慢了',
            '手机一般般 速度不快 反应迟钝4G用起来没有其他手机用的速度快，还有就是那些他送的东西没有一个有用的都是什么垃圾来的 你们看看图片上的手机壳 还有保护膜垃圾到根本不能用 一张塑胶片，贴上去整个屏幕都是泡泡 还有一个就是后面防摔的也是没用的 也是塑胶',
            '不错，，不错啊，，，',
            '宝贝很不错很喜欢～快递速度很快，客服人很好～物美价廉啊～值得推荐的～',
            '快递用了5天，实在觉得有点离谱',
            '这个必须给差评 东西贵不说吧，还要收我8块快递费，早知道就不在网上买这个了，而且快递还这么慢，18号下的单，到23号才收到，蜗牛快递吗？同天下的单 第二天就收到了，这个是比别的都特殊还是咋滴，服了',
            '不错，虽然没赶上，但是店家尽力了'
        ]

        # for str in texts:
        #     translation = translator.translate(str, dest='en')
        #     print(translation.text)

        i = 0
        with open('jd_reviews.csv', newline='') as csvfile:
            # spamreader = csv.reader(csvfile, delimiter=',')
            reader = csv.DictReader(csvfile)
            # fieldnames = reader.fieldnames
            for row in reader:
                with open('product.csv', 'a', newline='') as csvfile:
                    fieldnames = ['product_id', 'product_name', 'product_name_en']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
                    # writer.writeheader()
                    product_name_en = translator.translate(row['product_name'], dest='en').text
                    writer.writerow({
                        'product_id': row['product_id'],
                        'product_name': row['product_name'],
                        'product_name_en': product_name_en
                    })
                    i = i + 1
                    print('done with ', i)


process = PrepareData()
process.make_data()
print()
