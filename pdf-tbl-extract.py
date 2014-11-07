# -*- coding: utf-8 -*-

import cv2
from subprocess import Popen, PIPE, call
import tempfile
import numpy as np

pdf_name = "test/1077154006247_99651434705687.pdf"

jpg = [tempfile.mkdtemp(), "out"]

call("gs -dNOPAUSE -sDEVICE=jpeg -sOutputFile=" + jpg[0] + "/" + jpg[1] + "%d"
     + " -dJPEGQ=100 -q " + pdf_name + " -c quit", shell=True)

jpg_cur_file = jpg[0] + "/" + jpg[1] + str(1)


def shannon(g):
# Shannon entropy is a measure of how much information a series of values hold.
# If all the values have the same value, the Shannon entropy will be 0
# If the values span a wide range of numbers, the Shannon entropy will be high
    return -(lambda x: x != 0 and x*np.log2(x) or 0)(np.count_nonzero(g)/256.)

work = cv2.threshold(cv2.imread(
    jpg_cur_file, cv2.CV_LOAD_IMAGE_GRAYSCALE), 199, 1, cv2.THRESH_BINARY)[1]
hs = np.ones((len(work), len(work[0])), np.int8)
vs = hs.copy()
# We check the Shannon entropy of patches of the image.
# If it is 0 and also black, this might be a border
tbl_sep_len_pix = 100
# Start with patches from left to right
for row in range(work.shape[0]):
    for col in range(0, work.shape[1], tbl_sep_len_pix/10):
        sep = work[row:row + 1, col: col + tbl_sep_len_pix]
        if shannon(sep) < 0.01 and 0 == sep[0][0]:
            hs[row, col: col + tbl_sep_len_pix] &= 0
tbl_sep_len_pix = 20
# now patches from top to bottom
for col in range(work.shape[1]):
    for row in range(0, work.shape[0], tbl_sep_len_pix/10):
        sep = work[row: row + tbl_sep_len_pix, col: col + 1]
        if shannon(sep) < 0.01 and 0 == sep[0][0]:
            vs[row: row + tbl_sep_len_pix, col] &= 0
# Debug
#cv2.imwrite("test/hs.tiff", hs*255)
#cv2.imwrite("test/vs.tiff", vs*255)
tb = hs * vs
#cv2.imwrite("test/tb.tiff", tb*255)

#Let's determine where the table borders are
rsl = [hs[r, :].sum() / work.shape[1] for r in range(work.shape[0])]
csl = [vs[:, c].sum() / work.shape[0] for c in range(work.shape[1])]


def z(v):
    if 1 == z.last and v != 1:
        z.last = 0
        return True
    if 1 == v:
        z.last = 1
    return False

setattr(z, "last", 1)
rl = [r for r in range(work.shape[0]) if z(rsl[r])]
setattr(z, "last", 1)
cl = [c for c in range(work.shape[1]) if z(csl[c])]
# each td of table gets one entry here. 1 is standard,
# x means number of spans, 0 means that a preceeding td has span > 1
# handle cells spanning multiple columns
cs = [[np.all(tb[rl[r] + 1: rl[r + 1], cl[c + 1] - 1: cl[c + 1] + 1])
       for c in range(len(cl) - 1)] for r in range(len(rl) - 1)]
# handle cells spanning multiple rows
rs = [[np.all(tb[rl[r + 1] - 1: rl[r + 1] + 1, cl[c] + 1: cl[c + 1]])
       for r in range(len(rl) - 1)] for c in range(len(cl) - 1)]

# let's start writing the data
out_table = "<html><head><meta http-equiv='Content-Type' content='text/html; \
charset=UTF-8' /></head><style>td { border: 1px solid gray; }\
</style><body><table>\n"

rl_iter = iter(range(len(rl) - 1))
for i in rl_iter:
    out_table = out_table + "<tr>\n"
    cl_iter = iter(range(len(cl) - 1))
    for j in cl_iter:
        c_start = cl[j]
        r_start = rl[i]
        colspan = 1
        while j < (len(cl) - 1) and cs[i][j]:
            colspan += 1
            j = next(cl_iter)
        c_end = cl[j + 1]
        r_end = rl[i + 1]
        rowspan = 1
        if i > 0 and rs[j][i - 1]:
            continue
        else:
            t = j
            while t < (len(rl) - 1) and rs[t][i]:
                rowspan += 1
                t += 1
            r_end = rl[i + rowspan]
        # use pdftotext to extract the text.
        process = Popen("pdftotext "
                        + pdf_name + " -f 1 -l 1 -x " + str(c_start)
                        + " -y " + str(r_start)
                        + " -W " + str(c_end - c_start)
                        + " -H " + str(r_end - r_start)
                        + " -layout -nopgbrk -", shell=True, stdout=PIPE)
        content = process.stdout.read()
        out_table = out_table + "<td colspan='" + str(colspan) +\
            "' rowspan='" + str(rowspan) + "'>" + content + "</td>"
    out_table += "</tr>\n"
out_table += "</table></body></html>"

with open("out.html", "w") as out_file:
    out_file.write(out_table)
