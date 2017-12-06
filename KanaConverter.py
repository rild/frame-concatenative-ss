from pykakasi import kakasi

class KanaConverter:
    def __init__(self):
        self.kakasi = kakasi()

    def kana_to_roma(self, text):
        self.kakasi.setMode("K", "a")
        conv = self.kakasi.getConverter()
        return conv.do(text)
