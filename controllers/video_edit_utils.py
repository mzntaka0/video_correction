# -*- coding: shift-jis -*-
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals
import re
import datetime

def string_to_ffmpeg_format(string):
    pat = "(\d+)[^\d]+(\d+)"

    match = re.findall(pat, string=string)
    if len(match) != 1:
        return None

    integers = list(map(int, match[0]))

    if len(integers) != 2 or integers[0] >= integers[1]:
        return None

    start = integers[0]
    length = integers[1] - integers[0]
    start = str(datetime.timedelta(seconds=start))
    length = str(datetime.timedelta(seconds=length))

    return ('0' + start, '0' + length)


if __name__ == '__main__':
    ret = string_to_ffmpeg_format(u'15秒から32秒まで')
    print(ret)
