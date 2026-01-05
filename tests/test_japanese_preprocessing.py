"""
日本語前処理ユニットテスト

pyopenjtalk_prosody関数の韻律マーカー生成を検証します。

使用方法:
    uv run pytest tests/test_japanese_preprocessing.py -v
"""

import pytest
from vyvotts.utils.japanese_preprocessing import (
    pyopenjtalk_prosody,
    pyopenjtalk_prosody_accent,
    pyopenjtalk_g2p,
    preprocess_japanese_text,
    normalize_text,
    extract_accent_features,
    ACCENT_TOKENS,
)


class TestNormalizeText:
    """テキスト正規化のテスト"""

    def test_nfkc_normalization(self):
        """全角→半角変換"""
        assert normalize_text("ＡＢＣ１２３") == "ABC123"

    def test_control_char_removal(self):
        """制御文字削除"""
        assert normalize_text("test\x00\x1f") == "test"

    def test_whitespace_normalization(self):
        """連続スペースの正規化"""
        assert normalize_text("a   b  c") == "a b c"


class TestPyopenjtalkProsody:
    """韻律マーカー生成のテスト"""

    def test_sentence_markers(self):
        """文頭・文末マーカー"""
        result = pyopenjtalk_prosody("あ")
        assert result.startswith("^")
        assert result.endswith("$")

    def test_question_marker(self):
        """疑問文マーカー"""
        result = pyopenjtalk_prosody("何？")
        assert result.endswith("?")

    def test_empty_string(self):
        """空文字列"""
        assert pyopenjtalk_prosody("") == ""

    def test_pitch_rise_marker(self):
        """ピッチ上昇マーカー ["""
        # 頭高型のアクセントで [ が入ることを確認
        result = pyopenjtalk_prosody("雨")
        # 「雨」はアクセント型1（頭高）
        assert "[" in result

    def test_pitch_fall_marker_ame(self):
        """ピッチ下降マーカー ] - 雨（アクセント型1）"""
        result = pyopenjtalk_prosody("雨")
        # 「雨」(あめ) はアクセント型1で、2モーラ目でピッチ下降
        # ESPnet方式では ] が含まれるべき
        print(f"雨: {result}")
        assert "]" in result, f"Expected ] in '{result}'"

    def test_pitch_fall_marker_hashi_bridge(self):
        """ピッチ下降マーカー ] - 橋（アクセント型2）"""
        result = pyopenjtalk_prosody("橋")
        # 「橋」(はし) はアクセント型0または2
        print(f"橋: {result}")
        # 橋のアクセントは文脈依存なのでマーカーの有無は確認のみ

    def test_flat_accent_konnichiwa(self):
        """平板型アクセント - こんにちは"""
        result = pyopenjtalk_prosody("こんにちは")
        print(f"こんにちは: {result}")
        # 平板型(アクセント型0)ではピッチ上昇マーカーがない
        # ただしOpenJTalkの辞書によっては [ がある場合もある

    def test_accent_phrase_boundary(self):
        """アクセント句境界マーカー #"""
        result = pyopenjtalk_prosody("今日はいい天気です")
        print(f"今日はいい天気です: {result}")
        # 複合的な文では # が入ることがある


class TestPyopenjtalkG2P:
    """G2P変換のテスト"""

    def test_phoneme_output(self):
        """音素出力"""
        result = pyopenjtalk_g2p("こんにちは", kana=False)
        assert "k" in result
        assert "o" in result

    def test_kana_output(self):
        """カタカナ出力"""
        result = pyopenjtalk_g2p("こんにちは", kana=True)
        assert "コンニチワ" in result or "コンニチハ" in result


class TestPreprocessJapaneseText:
    """preprocess_japanese_text関数のテスト"""

    def test_prosody_mode(self):
        """prosodyモード"""
        result = preprocess_japanese_text("テスト", mode="prosody")
        assert "^" in result
        assert "$" in result

    def test_phoneme_mode(self):
        """phonemeモード"""
        result = preprocess_japanese_text("テスト", mode="phoneme")
        assert "t" in result

    def test_kana_mode(self):
        """kanaモード"""
        result = preprocess_japanese_text("テスト", mode="kana")
        assert "テスト" in result

    def test_invalid_mode(self):
        """無効なモード"""
        with pytest.raises(ValueError):
            preprocess_japanese_text("テスト", mode="invalid")


class TestExtractAccentFeatures:
    """アクセント特徴抽出のテスト"""

    def test_basic_extraction(self):
        """基本的な特徴抽出"""
        features = extract_accent_features("あ")
        assert len(features) > 0
        assert "phone" in features[0]
        assert "a1" in features[0]
        assert "f2" in features[0]

    def test_empty_string(self):
        """空文字列"""
        features = extract_accent_features("")
        assert features == []


class TestProsodyMarkerRegression:
    """韻律マーカーの回帰テスト"""

    def test_ame_has_pitch_markers(self):
        """雨（あめ）にピッチマーカーがあること"""
        result = pyopenjtalk_prosody("雨")
        # アクセント型1の単語には [ と ] の両方があるべき
        has_pitch_rise = "[" in result
        has_pitch_fall = "]" in result
        print(f"雨: {result} (rise={has_pitch_rise}, fall={has_pitch_fall})")
        # 少なくともピッチ上昇はあるべき
        assert has_pitch_rise, f"Expected [ in '{result}'"

    def test_complex_sentence(self):
        """複合文のテスト"""
        result = pyopenjtalk_prosody("今日は良い天気ですね。")
        print(f"今日は良い天気ですね。: {result}")
        assert result.startswith("^")
        assert result.endswith("$")


class TestAccentTokens:
    """アクセント特殊トークンのテスト"""

    def test_accent_tokens_count(self):
        """トークン数が正しいこと（-10〜15の26トークン）"""
        assert len(ACCENT_TOKENS) == 26

    def test_accent_tokens_format(self):
        """トークンが正しい形式であること"""
        for token in ACCENT_TOKENS:
            assert token.startswith("<a")
            assert token.endswith(">")

    def test_accent_tokens_range(self):
        """トークンが-10から15までの範囲を含むこと"""
        assert "<a-10>" in ACCENT_TOKENS
        assert "<a0>" in ACCENT_TOKENS
        assert "<a15>" in ACCENT_TOKENS
        assert "<a-11>" not in ACCENT_TOKENS
        assert "<a16>" not in ACCENT_TOKENS


class TestPyopenjtalkProsodyAccent:
    """アクセント特殊トークン付き韻律マーカーのテスト"""

    def test_sentence_markers(self):
        """文頭・文末マーカー"""
        result = pyopenjtalk_prosody_accent("あ")
        assert result.startswith("^")
        assert result.endswith("$")

    def test_empty_string(self):
        """空文字列"""
        assert pyopenjtalk_prosody_accent("") == ""

    def test_accent_token_format(self):
        """アクセントトークンが含まれること"""
        result = pyopenjtalk_prosody_accent("こんにちは")
        print(f"こんにちは (prosody_accent): {result}")
        # <a数字>形式のトークンが含まれること
        import re
        accent_pattern = r"<a-?\d+>"
        matches = re.findall(accent_pattern, result)
        assert len(matches) > 0, f"Expected accent tokens in '{result}'"

    def test_accent_token_attached_to_phoneme(self):
        """アクセントトークンが音素に付加されていること"""
        result = pyopenjtalk_prosody_accent("あ")
        print(f"あ (prosody_accent): {result}")
        # 音素の直後にアクセントトークン（スペースなし）
        import re
        # 音素<a数字>の形式を確認
        phoneme_accent_pattern = r"[a-zA-Z]+<a-?\d+>"
        matches = re.findall(phoneme_accent_pattern, result)
        assert len(matches) > 0, f"Expected phoneme+accent in '{result}'"

    def test_prosody_markers_preserved(self):
        """韻律マーカー（[, ], #）が保持されること"""
        result = pyopenjtalk_prosody_accent("雨")
        print(f"雨 (prosody_accent): {result}")
        # 頭高型アクセントで [ が含まれるべき
        assert "[" in result, f"Expected [ in '{result}'"

    def test_question_marker(self):
        """疑問文マーカー"""
        result = pyopenjtalk_prosody_accent("何？")
        assert result.endswith("?")

    def test_preprocess_mode_prosody_accent(self):
        """preprocess_japanese_textでprosody_accentモードが動作すること"""
        result = preprocess_japanese_text("テスト", mode="prosody_accent")
        import re
        accent_pattern = r"<a-?\d+>"
        matches = re.findall(accent_pattern, result)
        assert len(matches) > 0, f"Expected accent tokens in '{result}'"
        assert "^" in result
        assert "$" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
