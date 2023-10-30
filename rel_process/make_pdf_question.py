#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : make_pdf_question.py
# @Author: 罗锦文
# @Date  : 2023/9/12
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import pandas as pd
def read_question():
    ins = pd.read_csv('data/pdf_mrc.csv', header=None)
    queryset = set()
    for idx, v in enumerate(ins.values):
        if idx == 0:
            continue
        query = v[13]
        start = query.find('send prompt [')
        end = query.find('] to llm')
        query = query[start+len('send prompt ['):end]
        queryset.add(query)
    return queryset

def search_question():
    ins = pd.read_csv('data/pdf_query.csv', header=None)
    queryset = set()
    for idx, v in enumerate(ins.values):
        if idx == 0:
            continue
        query = v[13]
        # print(query)
        start = query.find('get answer ')
        query = query[start+len('get answer '):]
        query = eval(query)
        for a in query[0]:
            queryset.add(a)
    for a in queryset:
        print(a)
    return queryset

def get_resp():
    for line in sys.stdin:
        ins = json.loads(line)
        print(ins['query'].replace('\t', ' ').replace('\n', '<n>'), ins['response'].replace('\t', ' ').replace('\n', '<n>'), sep='\t')

if __name__ == "__main__":
    import json
    get_resp()
    # queryset = read_question()
    # prompt = """其中第1部分文章内容是：帮小忙阅读助手使用指引（内测版） 欢迎加入帮小忙阅读助手内测，我们在此为您准备了一份快速上手指南～祝您使用愉快。*注：本文仅作引导，内容有限，建议您打开自己的 PDF，解锁更丰富的解读功能哦～ 功能介绍 “帮小忙 PDF 阅读助手” 是 QQ 浏览器上线的新功能。您可通过与阅读助手对话，快速 获取文档大意、理解文章信息、掌握文档的核心及细节内容。 如何获取 PDF 文档 1. 通过「PDF 文件搜索」功能获取全网 PDF 若您没有 PDF 文档，或想获取更多有用有趣的 PDF 文档，可前往 https://tool.browser.qq.com/pdf_search.html 获取。 2. 通过「格式转换」功能，将已有 word 文档转为 PDF 您可先打开一份 word/ppt，点击下方工具栏中的「格式转换」，将文档转换为 PDF 格式后打开进行解读。 帮小忙功能使用说明 在「文件」页面中打开一篇 PDF 文档，点击右上角「 」按键，开始与帮小忙对话吧～ 当前帮小忙阅读助手支持以下功能： 1. 总结全文大意，快速获得摘要： 提问示例：”本文有什么值得提炼的信息？“ 阅读助手：“这篇文章介绍了 QQ 浏览器新推出的“帮小忙 PDF 阅读助手”功能，用户可以通过与该助手对话来快速获取文章大意、理解文章信息、掌握文章核心和细节内容。该助手支持总结全文大意、讨论选中的片段/词语以及开放性问题的探讨等功能。同时，提醒读者注意 AI 生成的回答仅供参考，不代表 QQ 浏览器的意见和立 场。” 2. 支持多轮对话，深入探讨： 提问示例：“刚才说的 AI 生成内容仅供参考，能否展开讲讲？”、“继续”、“还有 吗？” 阅读助手：“在这篇文章中提到，‘阅读助手的回复是由 AI 生成，并不代表 QQ 浏览器团队的观点和立场，只是供读者参考‘。因此，对 AI 的回答需要加以辨别，使用时需谨慎考虑。” 3. 支持选中片段/词语进行解读： 操作示例：选中「QQ 浏览器」这几个文字后，点击跳出的「帮我解读」键 阅读助手：“QQ 浏览器是一款流行的网页浏览软件，具有许多实用的功能和特性，例如快速加载和显示网页、内置广告拦截器、安全隐私保护等等。它还提供多种附加服务，如文件下载和管理、在线阅读和社交媒体分享等。” 4. 支持开放性问题探讨，更多能力等您解锁 提问示例：“阅读助手这个功能，它好用吗？” 阅读助手：“根据文章内容所述，帮小忙 PDF 阅读助手目前还处于测试阶段，因此无法确定其是否好用。但是，从文章中可以看到，此工具提供了多种功能，包括总结全文大意、深度解析文本段落、开放式问题解答等，这些都可以帮助用户更好地理解和处理 PDF 文件。如果您正在寻找一款好用的 PDF 阅读辅助工具，不妨尝试一下“帮小忙 PDF 阅读助手”。” *以上示例均来自阅读助手真实回答内容。 其他实用工具功能推荐 在您使用帮小忙解读 PDF 文档的过程中，总会遇到各类文件处理场景。我们为您推荐以下工具，希望可以在需要时帮到您～ 1. 文档格式互转 在 QQ 浏览器-「文件」-「PDF 工具」选择 PDF 转换功能，或使用 PDF 里下方功能栏中的「格式转换」功能，就能实现 PDF 文件与其他格式的自由转换。 • 图片转 PDF：选择手机相机图片的一张/多张图片，即可快速导出为 PDF 文件。 • 扫描转 PDF：用摄像头拍摄照片，系统自动把照片内容扫描成扫描件，可整合一 张/多张扫描照片保存为一个 PDF 文件。 • PDF 转多格式：使用「PDF 转长图」、「PDF 逐页转图」，即可把 PDF 快速生成一 张长图/PDF 页面逐页生成图片，方便保存以图片形式。使用「PDF 转 Word」、 「PDF 转 Excel」、「PDF 转 PPT」，就能把 PDF 文件转换成对应的文档格式。 2. 合并、翻译、瘦身 遇到 PDF 需要翻译文件的情况，该一页页截图去翻译软件翻译，还是转换格式再翻译？想整合多份 PDF 文件成一份 PDF，有没有更方便快捷的方法？ PDF 文件太大，怎么样才能让它小一点？ 大家的这些问题，QQ 浏览器都已经为你解决了！在 PDF 中「全部工具」-「特色工具」选项卡里「合并 PDF」、「文档翻译」、「PDF 密码」，和 QQ 浏览器-「文件」-「PDF 工具」的「PDF 瘦身」，就能找到这些特色又个性化的 PDF 功能，帮助你更好、更安全地处理 PDF 工作！ • 合并 PDF：选择要合并的多个 PDF 文件，预览并调整（页面顺序、旋转页面、删 除页面）新的 PDF 内容，一键保存生成一份合并后的 PDF 文件。 • 文档翻译：一键在线翻译 PDF 文件，实现英/日/韩语与中文的互译，可查看过往 PDF 文件的翻译记录。译文预览可输出翻译文件，双语对照无需切换页面查看，提高查看文件的效率，且不改变 PDF 原文内容的版式。 • PDF 瘦身：选择要瘦身的 PDF 文件，系统自动检测可瘦身内容减少文件所占存 储，一键导出一份瘦身后的 PDF 文件。 3. PDF 页面管理 在 PDF 的「全部工具」-「特色工具」选项卡里，选择「页面管理」工具，包括有提取、增加、旋转、删除的四种功能按钮，长按拖动某页内容即可调整 PDF 文档的页面顺序。该工具可以实现完成一个或多个 PDF 页面的处理。 • 提取：快速在该 PDF 文件中提取已选择的某一页/多页的内容，保存为一份新的 PDF 文档。 • 增加：在 PDF 文件中通过添加空白页、从相册添加、拍照添加、从 PDF 文档中添 加这四种内容添加方式，实现 PDF 增加指定内容、多个 PDF 部分内容合并的操作。 • 旋转：选择 PDF 文档的某一页/多页的内容，点击“旋转”按钮，页面内容进行 90°逆时针旋转，实现调整 PDF 指定页面的方向。 • 删除：可以快速删除 PDF 文件中的某一页/多张的内容。 • 顺序调整：在「页面管理」工具页面中，长按拖动一页内容，即可调整该内容在 PDF 文档中位置，从而实现 PDF 文档的页面顺序调整。 以上便是指引的全部内容啦～ 注意：阅读助手的回复将根据文档内容实时生成，并不代表 QQ 浏览器团队的意见与立场，仅供参考哦。\n根据上面内容回答问题：\n%s"""
    # for query in queryset:
    #     print(json.dumps({
    #         'prompt': prompt %query,
    #         'query': query
    #     }, ensure_ascii=False))
