"""
日本語TTS前処理ユーティリティ

pyopenjtalk-plusを使用して、日本語テキストを音声合成用に前処理します。
ESPNet/Style-BERT-VITS2で使用されているpyopenjtalk_prosody方式を採用。

使用方法:
    from vyvotts.utils.japanese_preprocessing import pyopenjtalk_prosody

    text = "こんにちは、今日はいい天気ですね。"
    prosody_text = pyopenjtalk_prosody(text)
    # → "^ k o [ N n i ch i w a # k y o o w a [ i i t e N k i d e s U n e $"
"""

import pyopenjtalk
import unicodedata
import re
from typing import List, Dict, Any, Optional


def normalize_text(text: str) -> str:
    """
    基本的なテキスト正規化を行う。

    Args:
        text: 入力テキスト

    Returns:
        正規化されたテキスト
    """
    # NFKC正規化（全角→半角、半角カナ→全角カナなど）
    text = unicodedata.normalize('NFKC', text)
    # 制御文字削除
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 連続スペースを1つに
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _numeric_feature_by_regex(regex: str, s: str) -> int:
    """
    正規表現で数値特徴を抽出する。

    Args:
        regex: 正規表現パターン
        s: 入力文字列

    Returns:
        抽出された数値、見つからない場合は-50
    """
    match = re.search(regex, s)
    if match:
        return int(match.group(1))
    return -50  # undefined


def extract_fullcontext_label(text: str) -> List[str]:
    """
    OpenJTalkのフルコンテキストラベルを抽出する。

    Args:
        text: 入力テキスト

    Returns:
        フルコンテキストラベルのリスト
    """
    return pyopenjtalk.extract_fullcontext(text)


def pyopenjtalk_prosody(text: str, drop_unvoiced_vowels: bool = True) -> str:
    """
    日本語テキストを韻律情報付き音素列に変換する（ESPNet方式）。

    韻律マーカー:
        ^ : 文頭
        $ : 文末（平叙文）
        ? : 文末（疑問文）
        # : アクセント句境界
        [ : ピッチ上昇
        ] : ピッチ下降

    Args:
        text: 入力テキスト
        drop_unvoiced_vowels: 無声母音を小文字に変換するか

    Returns:
        韻律マーカー付き音素列（スペース区切り）

    Example:
        >>> pyopenjtalk_prosody("こんにちは")
        "^ k o [ N n i ch i w a $"
    """
    # テキスト正規化
    text = normalize_text(text)

    if not text:
        return ""

    labels = extract_fullcontext_label(text)
    N = len(labels)

    if N == 0:
        return ""

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # 現在の音素を抽出
        p3_match = re.search(r"\-(.*?)\+", lab_curr)
        if not p3_match:
            continue
        p3 = p3_match.group(1)

        # 無音（pau/sil）はスキップ
        if p3 in ["sil", "pau"]:
            continue

        # 無声母音の処理（大文字→小文字）
        if drop_unvoiced_vowels and p3 in ["A", "I", "U", "E", "O"]:
            p3 = p3.lower()

        # 文頭マーカー（最初の非無音音素の前）
        if len(phones) == 0:
            phones.append("^")

        # A フィールド（モーラ・アクセント位置）を抽出
        a1 = _numeric_feature_by_regex(r"/A:(\-?[0-9]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # ピッチ上昇マーカー（アクセント句の先頭で、かつアクセント核の前）
        if a3 == 1 and a2 == 1:
            phones.append("[")

        phones.append(p3)

        # ピッチ下降マーカー（アクセント核の直後）
        if a1 == 0 and a2 == a3:
            phones.append("]")

        # アクセント句境界マーカー
        if n < N - 1:
            lab_next = labels[n + 1]
            # 次の音素がアクセント句の先頭かどうか
            p3_next_match = re.search(r"\-(.*?)\+", lab_next)
            if p3_next_match:
                p3_next = p3_next_match.group(1)
                if p3_next not in ["sil", "pau"]:
                    a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
                    if a2_next == 1 and a3 == 1:
                        phones.append("#")

    # 文末マーカー
    if phones:
        if text.endswith("?") or text.endswith("？"):
            phones.append("?")
        else:
            phones.append("$")

    return " ".join(phones)


def pyopenjtalk_g2p(text: str, kana: bool = False) -> str:
    """
    日本語テキストをG2P（Grapheme-to-Phoneme）変換する。

    Args:
        text: 入力テキスト
        kana: Trueならカタカナ、Falseなら音素列を返す

    Returns:
        変換後のテキスト

    Example:
        >>> pyopenjtalk_g2p("音声合成技術", kana=True)
        "オンセイゴーセイギジュツ"
        >>> pyopenjtalk_g2p("音声合成技術", kana=False)
        "o N s e i g o o s e i g i j u ts u"
    """
    text = normalize_text(text)
    if not text:
        return ""
    return pyopenjtalk.g2p(text, kana=kana)


def extract_accent_features(text: str) -> List[Dict[str, Any]]:
    """
    テキストからアクセント特徴を抽出する。

    Args:
        text: 入力テキスト

    Returns:
        各音素のアクセント情報を含む辞書のリスト

    Features:
        - phone: 音素
        - a1 (mora_diff_from_accent): アクセント核からのモーラ差
        - a2 (mora_pos_forward): アクセント句内モーラ位置（先頭から）
        - a3 (mora_pos_backward): アクセント句内モーラ位置（末尾から）
        - f1 (accent_phrase_moras): アクセント句のモーラ数
        - f2 (accent_type): アクセント型（0=平板、n=nモーラ目に核）
        - f3 (is_interrogative): 疑問形フラグ
    """
    text = normalize_text(text)
    if not text:
        return []

    labels = extract_fullcontext_label(text)
    features = []

    for label in labels:
        # 音素抽出
        phone_match = re.search(r"\-(.*?)\+", label)
        if not phone_match:
            continue
        phone = phone_match.group(1)

        if phone in ["sil", "pau"]:
            continue

        # A フィールド（モーラ・アクセント位置）
        a_match = re.search(r"/A:(\-?\d+)\+(\d+)\+(\d+)/", label)
        if a_match:
            a1 = int(a_match.group(1))
            a2 = int(a_match.group(2))
            a3 = int(a_match.group(3))
        else:
            a1, a2, a3 = 0, 0, 0

        # F フィールド（アクセント句特徴）
        f_match = re.search(r"/F:(\d+)_(\d+)#(\d+)", label)
        if f_match:
            f1 = int(f_match.group(1))
            f2 = int(f_match.group(2))
            f3 = int(f_match.group(3))
        else:
            f1, f2, f3 = 0, 0, 0

        features.append({
            'phone': phone,
            'a1': a1,  # mora_diff_from_accent
            'a2': a2,  # mora_pos_forward
            'a3': a3,  # mora_pos_backward
            'f1': f1,  # accent_phrase_moras
            'f2': f2,  # accent_type
            'f3': f3,  # is_interrogative
        })

    return features


def preprocess_japanese_text(
    text: str,
    mode: str = "prosody"
) -> str:
    """
    日本語テキストをTTS用に前処理する。

    Args:
        text: 入力テキスト
        mode: 変換モード
            - "prosody": 韻律マーカー付き音素列（最高品質、推奨）
            - "phoneme": 音素列のみ
            - "kana": カタカナ読み

    Returns:
        前処理済みテキスト

    Example:
        >>> preprocess_japanese_text("こんにちは", mode="prosody")
        "^ k o [ N n i ch i w a $"
        >>> preprocess_japanese_text("こんにちは", mode="phoneme")
        "k o N n i ch i w a"
        >>> preprocess_japanese_text("こんにちは", mode="kana")
        "コンニチワ"
    """
    if mode == "prosody":
        return pyopenjtalk_prosody(text)
    elif mode == "phoneme":
        return pyopenjtalk_g2p(text, kana=False)
    elif mode == "kana":
        return pyopenjtalk_g2p(text, kana=True)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'prosody', 'phoneme', or 'kana'.")


# 便利なエイリアス
g2p = pyopenjtalk_g2p
prosody = pyopenjtalk_prosody
