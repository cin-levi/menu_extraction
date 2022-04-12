import logging
import re
from typing import Callable, Optional

from pydantic import BaseModel

logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.INFO)


class Bullet(BaseModel):
    patterns = '➢･・●※◆•‣-⁃⁌⁍∙○◘◦⦾⦿􀂾◇'


class Numbering(BaseModel):
    patterns = {
        re.compile(r'^\(\d+\)'),
        re.compile(r'^\d{1,2}[.)、\s]'),
        re.compile(r'^第\d{1,2}章'),
        re.compile(r'^第\d{1,2}節'),
    }


class Header(BaseModel):
    patterns = [
        {
            re.compile(r'^第\d{1,2}章.{0,25}'),
            re.compile(r'^附則$'),
        },
        {
            re.compile(r'^第\d{1,2}節[ (]'),
            re.compile(r'^第\d{1,2}節$'),
        },
        {
            re.compile(r'^[(《【〈\[].*[)》】〉\]]$'),
            re.compile(r'^第.*\(.*\)'),
        },
        {
            re.compile(r'^第[\s\d]*条[^(]{2}.*'),
            re.compile(r'^第[\s\d]*条$'),
            re.compile(r'^第\d{1,2}条の\d+[ \[]'),
            re.compile(r'^第[ ]*\d{1,2}[ ]*項$'),
        },
        {
            re.compile(r"^別表第*\(*\d{1,3}\)*|^表\d{1,3}")
        }
    ]


class Debug(BaseModel):
    class BaseDebug(BaseModel):
        pages = []

    class PDFDebug(BaseDebug):
        gen_image = False

    enable = False
    location = '.debug'
    show_para = False
    show_tree = False

    pdf = PDFDebug()


class Structuring(BaseModel):
    from ..structuring import StructuringRule
    rules = [
        StructuringRule.FirstParagraphRule(),
        StructuringRule.ExpandedTableRule(),
        StructuringRule.TocRule(),
        StructuringRule.NumberingRule(),
        StructuringRule.BulletRule(),
        StructuringRule.TitleRule(),
        StructuringRule.IndentRule(),
    ]


class Constructing(BaseModel):
    rules = []


class Exporting(BaseModel):
    location = '.cache'
    cell_deliminator = '|'

class Labeling(BaseModel):
    location = '.labeling'

class Settings(BaseModel):
    class PDF(BaseModel):
        bullet = Bullet()
        numbering = Numbering()
        header = Header()

        tolerance_x = 20
        tolerance_y = 3
        table_settings = {}

        from ...pdf.rule import Rule
        constructing = Constructing(rules=[
            Rule.FirstLine(),
            Rule.BulletLine(),
            Rule.NumberingLine(),
            Rule.Title(),
            Rule.OppositeAlignment(),
            Rule.LineLength(),
            Rule.IndentationDifference(),
            # Rule.IndentationShorter(),
            Rule.DotEnding(),
            Rule.LineDistance(),
            Rule.ShortLine(),
            Rule.PositiveBaseRule()
        ])
        structuring = Structuring()

        line_preprocess: Optional[Callable] = None
        line_postprocess: Optional[Callable] = None


    debug = Debug()
    exporting = Exporting()
    labeling = Labeling()

    pdf = PDF()
    suffix_filter = {'.pdf', '.html', }


settings = Settings()
