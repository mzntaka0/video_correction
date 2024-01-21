import unittest
import controllers.video_edit_utils as video_edit_utils
from pykakasi import kakasi


class TestConvert(unittest.TestCase):
    def setUp(self):
        kakashi = kakasi()
        kakashi.setMode('H', 'a')
        kakashi.setMode('K', 'a')
        kakashi.setMode('J', 'a')
        self.conv = kakashi.getConverter()

    def test_string_to_ffmpeg_format(self):

        test_case = ["15秒から20秒", "40秒から20秒", "100から150", "重病から2重病", "20秒と40秒と60秒", "2040"]
        test_case_answer = [(15, 5), None, (100, 50), None, (20, 20), None]

        for in_string, assumed in zip(test_case, test_case_answer):
            kakashi_txt = self.conv.do(in_string)
            ret = video_edit_utils.string_to_ffmpeg_format(kakashi_txt)
            self.assertEqual(ret, assumed)


if __name__ == "__main__":
    unittest.main()
