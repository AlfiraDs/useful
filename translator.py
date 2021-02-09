# labels_encoding: utf-8

from googletrans import Translator
from mtranslate import translate
import pandas as pd
import time
from sklearn.datasets import fetch_20newsgroups


class TextTranslator:
    def __init__(self, src_file, dst_file, src_lang="auto", dst_lang='en'):
        """
        Translates texts in src_file and saves translation to dst_file. It also keeps on track what was translated, so
        it can be translated in multiply runs.
        :param src_file: path to file with texts (one column csv without a header)
        :param dst_file:
        :param dst_lang: one of googletrans.LANGUAGES
        """
        self.translator = Translator(
            ["translate.google.com"]
        )
        self.start_time = time.time()
        self.src_file = src_file
        self.dst_file = dst_file
        self.dst_lang = dst_lang
        self.src_lang = src_lang
        self.src_texts = self.load_src()
        self.dst_df = self.load_dst()

    def load_src(self):
        return pd.read_csv(self.src_file, encoding='utf-8', squeeze=True, header=None, quotechar='"')

    def load_dst(self):
        try:
            return pd.read_csv(self.dst_file, encoding='utf-8', squeeze=True, header=None, quotechar='"')
        except IOError:
            return pd.Series()

    def save(self):
        self.dst_df.to_csv(self.dst_file, encoding='utf-8', index=False, header=False)
        print('saved ', self.dst_df.shape[0], 'of', self.src_texts.shape[0])

    def translate(self):
        start, end = self.dst_df.shape[0], self.src_texts.shape[0]
        for idx in range(start, end):
            text = self.src_texts[idx]
            self.dst_df.loc[idx] = self.translate_text(text)
            self.save()
        if start == end:
            print('there is nothing to translate')

    def translate_text(self, text):
        wait_after = 1 * 60
        waiting_time = 0.5 * 60
        if time.time() - self.start_time >= wait_after:
            print('waiting', waiting_time, 'sec and creating new translator')
            time.sleep(waiting_time)
            self.start_time = time.time()
            self.translator = Translator()
        # return self.translator.translate(text, src=self.src_lang, dest=self.dst_lang).text
        return translate(text, from_language=self.src_lang, to_language=self.dst_lang)


if __name__ == "__main__":
    src = pd.Series(fetch_20newsgroups().data[:10]).to_csv("src.csv", encoding='utf-8', index=None, header=None)
    TextTranslator("src.csv", "dst.csv", src_lang="auto", dst_lang="es").translate()
