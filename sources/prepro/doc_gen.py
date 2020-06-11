# coding: utf-8
import os, glob

def listup_files():
  return glob.glob("./texts/*.txt")

def replacer(txt):
  txt = txt.replace("：", "").replace("≫", "").replace("「", "").replace("」", "").replace(".", "").replace("、", "").replace("　", "").replace("？", "").replace("！", "").replace("♬", "").replace("〜", "").replace(" ", "").replace("＞", "").replace("＜", "").replace("…", "").replace("『", "").replace("』", "").replace("［", "").replace("］", "").replace("〈", "").replace("〉", "").replace("《", "").replace("》", "")

  txt = txt.replace("（", "").replace("）", "").replace("『", "").replace("』", "").replace("→", "").replace("☆", "").replace("〈", "").replace("〉", "").replace("・", "").replace("⁈", "").replace("♫", "")

  txt = txt.replace('⛌', '').replace('⛣', '').replace('➡', '').replace('㈪', '').replace('Ⅰ', '').replace('⛍', '').replace('⭖', '').replace('⬅', '').replace('㈫', '').replace('Ⅱ', '').replace('❗', '').replace('⭗', '').replace('⬆', '').replace('㈬', '').replace('Ⅲ', '').replace('⛏', '').replace('⭘', '').replace('⬇', '').replace('㈭', '').replace('Ⅳ', '').replace('⛐', '').replace('⭙', '').replace('⬯', '').replace('㈮', '').replace('Ⅴ', '').replace('⛑', '').replace('☓', '').replace('⬮', '').replace('㈯', '').replace('Ⅵ', '').replace('◻', '').replace('㊋', '').replace('年', '').replace('㈰', '').replace('Ⅶ', '').replace('⛒', '').replace('〒', '').replace('月', '').replace('㈷', '').replace('Ⅷ', '').replace('⛕', '').replace('⛨', '').replace('日', '').replace('㍾', '').replace('Ⅸ', '').replace('⛓', '').replace('㉆', '').replace('円', '').replace('㍽', '').replace('Ⅹ', '').replace('⛔', '').replace('㉅', '').replace('㎡', '').replace('㍼', '').replace('Ⅺ', '').replace('◻', '').replace('⛩', '').replace('㎥', '').replace('㍻', '').replace('Ⅻ', '').replace('◻', '').replace('࿖', '').replace('㎝', '').replace('№', '').replace('⑰', '').replace('◻', '').replace('⛪', '').replace('㎠', '').replace('℡', '').replace('⑱', '').replace('◻', '').replace('⛫', '').replace('㎤', '').replace('〶', '').replace('⑲', '').replace('🅿', '').replace('⛬', '').replace('🄀', '').replace('⚾', '').replace('⑳', '').replace('🆊', '').replace('♨', '').replace('⒈', '').replace('🉀', '').replace('◻', '').replace('◻', '').replace('⛭', '').replace('⒉', '').replace('🉁', '').replace('◻', '').replace('◻', '').replace('⛮', '').replace('⒊', '').replace('🉂', '').replace('◻', '').replace('⛖', '').replace('⛯', '').replace('⒋', '').replace('🉃', '').replace('◻', '').replace('⛗', '').replace('⚓', '').replace('⒌', '').replace('🉄', '').replace('◻', '').replace('⛘', '').replace('✈', '').replace('⒍', '').replace('🉅', '').replace('◻', '').replace('⛙', '').replace('⛰', '').replace('⒎', '').replace('🉆', '').replace('◻', '').replace('⛚', '').replace('⛱', '').replace('⒏', '').replace('🉇', '').replace('◻', '').replace('⛛', '').replace('⛲', '').replace('⒐', '').replace('🉈', '').replace('◻', '').replace('⛜', '').replace('⛳', '').replace('氏', '').replace('🄪', '').replace('◻', '').replace('⛝', '').replace('⛴', '').replace('副', '').replace('🈧', '').replace('◻', '').replace('⛞', '').replace('⛵', '').replace('元', '').replace('🈨', '').replace('◻', '').replace('⛟', '').replace('🅗', '').replace('故', '').replace('🈩', '').replace('◻', '').replace('⛠', '').replace('Ⓓ', '').replace('前', '').replace('🈔', '').replace('◻', '').replace('⛡', '').replace('Ⓢ', '').replace('新', '').replace('🈪', '').replace('◻', '').replace('⭕', '').replace('⛶', '').replace('🄁', '').replace('🈫', '').replace('◻', '').replace('㉈', '').replace('🅟', '').replace('🄂', '').replace('🈬', '').replace('🄐', '').replace('㉉', '').replace('🆋', '').replace('🄃', '').replace('🈭', '').replace('🄑', '').replace('㉊', '').replace('🆍', '').replace('🄄', '').replace('🈮', '').replace('🄒', '').replace('㉋', '').replace('🆌', '').replace('🄅', '').replace('🈯', '').replace('🄓', '').replace('㉌', '').replace('🅹', '').replace('🄆', '').replace('🈰', '').replace('🄔', '').replace('㉍', '').replace('⛷', '').replace('🄇', '').replace('🈱', '').replace('🄕', '').replace('㉎', '').replace('⛸', '').replace('🄈', '').replace('ℓ', '').replace('🄖', '').replace('㉏', '').replace('⛹', '').replace('🄉', '').replace('㎏', '').replace('🄗', '').replace('◻', '').replace('⛺', '').replace('🄊', '').replace('㎐', '').replace('🄘', '').replace('◻', '').replace('🅻', '').replace('㈳', '').replace('㏊', '').replace('🄙', '').replace('◻', '').replace('☎', '').replace('㈶', '').replace('㎞', '').replace('🄚', '').replace('◻', '').replace('⛻', '').replace('㈲', '').replace('㎢', '').replace('🄛', '').replace('⒑', '').replace('⛼', '').replace('㈱', '').replace('㍱', '').replace('🄜', '').replace('⒒', '').replace('⛽', '').replace('㈹', '').replace('◻', '').replace('🄝', '').replace('⒓', '').replace('⛾', '').replace('㉄', '').replace('◻', '').replace('🄞', '').replace('🅊', '').replace('🅼', '').replace('▶', '').replace('½', '').replace('🄟', '').replace('🅌', '').replace('⛿', '').replace('◀', '').replace('↉', '').replace('🄠', '').replace('🄿', '').replace('◻', '').replace('〖', '').replace('⅓', '').replace('🄡', '').replace('🅆', '').replace('◻', '').replace('〗', '').replace('⅔', '').replace('🄢', '').replace('🅋', '').replace('◻', '').replace('⟐', '').replace('¼', '').replace('🄣', '').replace('🈐', '').replace('◻', '').replace('²', '').replace('¾', '').replace('🄤', '').replace('🈑', '').replace('◻', '').replace('³', '').replace('⅕', '').replace('🄥', '').replace('🈒', '').replace('◻', '').replace('🄭', '').replace('⅖', '').replace('🄦', '').replace('🈓', '').replace('◻', '').replace('◻', '').replace('⅗', '').replace('🄧', '').replace('🅂', '').replace('◻', '').replace('◻', '').replace('⅘', '').replace('🄨', '').replace('🈔', '').replace('◻', '').replace('◻', '').replace('⅙', '').replace('🄩', '').replace('🈕', '').replace('◻', '').replace('◻', '').replace('⅚', '').replace('㉕', '').replace('🈖', '').replace('◻', '').replace('◻', '').replace('⅐', '').replace('㉖', '').replace('🅍', '').replace('◻', '').replace('◻', '').replace('⅛', '').replace('㉗', '').replace('🄱', '').replace('◻', '').replace('◻', '').replace('⅑', '').replace('㉘', '').replace('🄽', '').replace('◻', '').replace('◻', '').replace('⅒', '').replace('㉙', '').replace('⬛', '').replace('◻', '').replace('◻', '').replace('☀', '').replace('㉚', '').replace('⬤', '').replace('◻', '').replace('◻', '').replace('☁', '').replace('①', '').replace('🈗', '').replace('◻', '').replace('◻', '').replace('☂', '').replace('②', '').replace('🈘', '').replace('◻', '').replace('◻', '').replace('⛄', '').replace('③', '').replace('🈙', '').replace('◻', '').replace('◻', '').replace('☖', '').replace('④', '').replace('🈚', '').replace('◻', '').replace('◻', '').replace('☗', '').replace('⑤', '').replace('🈛', '').replace('◻', '').replace('◻', '').replace('⛉', '').replace('⑥', '').replace('⚿', '').replace('◻', '').replace('◻', '').replace('⛊', '').replace('⑦', '').replace('🈜', '').replace('◻', '').replace('◻', '').replace('♦', '').replace('⑧', '').replace('🈝', '').replace('◻', '').replace('◻', '').replace('♥', '').replace('⑨', '').replace('🈞', '').replace('◻', '').replace('◻', '').replace('♣', '').replace('⑩', '').replace('🈟', '').replace('◻', '').replace('◻', '').replace('♠', '').replace('⑪', '').replace('🈠', '').replace('◻', '').replace('◻', '').replace('⛋', '').replace('⑫', '').replace('🈡', '').replace('◻', '').replace('◻', '').replace('⨀', '').replace('⑬', '').replace('🈢', '').replace('◻', '').replace('◻', '').replace('‼', '').replace('⑭', '').replace('🈣', '').replace('◻', '').replace('◻', '').replace('⁈', '').replace('⑮', '').replace('🈤', '').replace('◻', '').replace('◻', '').replace('⛅', '').replace('⑯', '').replace('🈥', '').replace('◻', '').replace('◻', '').replace('☔', '').replace('❶', '').replace('🅎', '').replace('◻', '').replace('◻', '').replace('⛆', '').replace('❷', '').replace('㊙', '').replace('◻', '').replace('◻', '').replace('☃', '').replace('❸', '').replace('🈀', '').replace('◻', '').replace('◻', '').replace('⛇', '').replace('❹', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('⚡', '').replace('❺', '').replace('◻', '').replace('◻', '').replace('🄬', '').replace('⛈', '').replace('❻', '').replace('◻', '').replace('◻', '').replace('🄫', '').replace('◻', '').replace('❼', '').replace('◻', '').replace('◻', '').replace('㉇', '').replace('⚞', '').replace('❽', '').replace('◻', '').replace('◻', '').replace('🆐', '').replace('⚟', '').replace('❾', '').replace('◻', '').replace('◻', '').replace('🈦', '').replace('♫', '').replace('❿', '').replace('◻', '').replace('◻', '').replace('℻', '').replace('☎', '').replace('⓫', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('⓬', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('㉛', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('◻', '').replace('◻', '')
  return txt

def doc_gen():
  for p in listup_files():
    with open(str(p), "r") as f:
      with open("document_set.txt", "a") as ff:
        ff.write(replacer(f.read()).replace("。", ".") + "\n")

if __name__ == "__main__":
  doc_gen()
