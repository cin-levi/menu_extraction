from pathlib import Path
import PyPDF2
import os
import platform
import re
import shutil
from subprocess import PIPE, Popen
import tempfile
import glob
import cv2


def _get_command_path(command):
    """get command path"""
    if platform.system() == "Windows":
        command = command + ".exe"
    return command


def _pdf_info(pdf_path):
    command = [_get_command_path("pdfinfo"), pdf_path]
    proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=False)
    out = proc.communicate()[0]

    page_count = int(
        re.search(r"Pages:\s+(\d+)", out.decode("utf8", "ignore")).group(1)
    )
    pdf_info = []
    for i in range(page_count):
        cmd = [
            _get_command_path("pdfinfo"),
            "-f",
            str(i + 1),
            "-l",
            str(i + 1),
            pdf_path,
        ]
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        out = proc.communicate()[0]
        # Page size: 792 x 1224 pts
        page_size = (
            re.search(r"Page.*size:\s+([\d\.x\s]+)",
                      out.decode("utf8", "ignore"))
                .group(1)
                .split("x")
        )
        pdf_info.append((str(i + 1), list(map(float, page_size))))

    return pdf_info


def pdf_to_image(pdf_path="./tmp/hihi/GED.pdf", image_dir='tests/.cache',
                 thread_num=6, dpi=300):
    """Convert pdf file to image(s)"""
    assert "pdf" in pdf_path
    page_nums = [pnum for (pnum, _) in _pdf_info(pdf_path)]
    print("page_nums: ", page_nums)

    processes = []

    for page_num in page_nums:
        command = [
            _get_command_path("pdftoppm"),
            "-png",
            "-singlefile",
            "-r",
            str(dpi),
            "-f",
            page_num,
            "-l",
            page_num,
            pdf_path,
            os.path.join(image_dir, page_num),
        ]
        processes.append(Popen(command, stdout=PIPE, stderr=PIPE))

        if len(processes) == thread_num:
            processes[0].wait()
            processes.pop(0)

    for proc in processes:
        proc.wait()
    return image_dir


def _convert_pdf_to_image(pdf_path) -> Path:
    """ Convert a pdf to images, an image for a page, all under the same
    directory which name from the hash of the input file path
    :param pdf_path: path to a pdf file
    :return: path to the images dir
    """
    pdf_path = Path(pdf_path)
    name = pdf_path.name
    with tempfile.TemporaryDirectory() as tmpdir:
        print("tmpdir: ", tmpdir)
        clean_pdf_path = tmpdir + '/' + name
        with open(pdf_path, 'rb') as f_in:
            pdf_in = PyPDF2.PdfFileReader(f_in)
            pdf_out = PyPDF2.PdfFileWriter()
            for page in pdf_in.pages:
                pdf_out.addPage(page)
            pdf_out.removeLinks()
            with open(clean_pdf_path, 'wb') as f_out:
                pdf_out.write(f_out)
        print("clean_pdf_path: ", clean_pdf_path)
        files = glob.glob(tmpdir + '/*')
        print(files)
        # write image
        pdf_to_image(clean_pdf_path,
                     image_dir=tmpdir,
                     dpi=100
                     )

        images = glob.glob(tmpdir + '/*.png')
        if len(images) > 0:
            img = cv2.imread(images[0])
            return img


if __name__ == '__main__':
    pdf_file = 'D:\\invoice-extraction\\data\\invoice_pdf\\test\\segment_5\\template_65.pdf'
    _convert_pdf_to_image(pdf_file)
    pass
