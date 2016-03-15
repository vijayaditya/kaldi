#!/usr/bin/env python


# Copyright 2016 Vijayaditya Peddinti.
# Apache 2.0.
import sys
import warnings
import imp
import argparse
import os
import errno
import logging
import re
import subprocess
import numpy
from random import shuffle

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    #from matplotlib import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    plot = True
except ImportError:
    warnings.warn("""
This script requires matplotlib and numpy. Please install them to generate plots. Proceeding with generation of tables.
If you are on a cluster where you do not have admin rights you could try using virtualenv.""")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Generating plots')




def GetArgs():
    parser = argparse.ArgumentParser(description="""
Parses the training logs and generates a variety of plots.
example : steps/nnet3/report/generate_plots.py --comparison-dir exp/nnet3/tdnn1 --comparison-dir exp/nnet3/tdnn2 exp/nnet3/tdnn exp/nnet3/tdnn/report
""")
    parser.add_argument("--key", type=str, required=True, action='append', help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--ali-file", type=str, required=True, action='append', help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--wav-dir", type=str, default=None, action='append', help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--utt-name-file", type=str, default=None)
    parser.add_argument("--max-plots", type=int, default=100, help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--text-file", type=str, default=None, help="other experiment directories for comparison. These will only be used for plots, not tables")
    parser.add_argument("--phone-file", type=str, default=None, help="other experiment directories for comparison. These will only be used for plots, not tables")

    parser.add_argument("output_dir", help="experiment directory, e.g. exp/nnet3/tdnn/report")

    args = parser.parse_args()

    if args.wav_dir is not None:
        assert(len(args.ali_file) == len(args.wav_dir))

    comparison_dict = {}
    for i in range(len(args.key)):
        comparison_dict[args.key[i]]={'ali_file':args.ali_file[i],
                              'wav_dir':args.wav_dir[i] if args.wav_dir is not None else None}
    args.ali_wav = comparison_dict
    return args

plot_colors = ['red', 'blue', 'green', 'black', 'magenta', 'yellow', 'cyan' ]

class LatexReport:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        self.document=[]
        self.document.append("""
%\documentclass{beamer}
\documentclass[envcountsect]{beamer}

\usepackage{multimedia}


\mode<presentation>
{
% \usetheme{Berlin}
  % or ...
%\usetheme{default}
%\usetheme{PaloAlto}
%\usetheme{Malmoe}
%\usetheme{Warsaw}
  \setbeamercovered{transparent}
  % or whatever (possibly just delete it)
\usecolortheme{wolverine}
%\usefonttheme{serif}
%\setbeamertemplate{blocks}[rounded][shadow=true]
%\usecolortheme{sidebartab}

%\setbeamercolor{some beamer element}{fg=red}
}
\\beamertemplateballitem

\usepackage{color}
%\usepackage[english]{babel}
% or whatever

%\usepackage[latin1]{inputenc}
% or whatever

%\usepackage{helvetica}
\usepackage{accents}
\usepackage{hyperref}

\usepackage{times}
\usepackage[T1]{fontenc}
% Or whatever. Note that the encoding and the font should match. If T1
% does not look nice, try deleting the line with the fontenc.

\\newcommand{\ul}{\underline}
\\newcommand{\\vx}{\\bf x}
\\newcommand{\la}{\lambda}
\\newcommand{\e}{{\\rm e}}


\\newtheorem{prop}[theorem]{Proposition}
\\newtheorem{quest}{Question}
\\newtheorem{remark}{Remark}

\setbeamerfont{caption}{series=\\normalfont,size=\\fontsize{5}{10}}

\\title
{Alignment issues in AMI data}
\\author[]{Vijay}


\\beamerdefaultoverlayspecification{<+->}

\\begin{document}

\\begin{frame}
%\sound[autostart, inlinesound]{}{show.wav}
  \\titlepage
\end{frame}

\section{1. Alignments}

""")

    def AddFigure(self, figure_pdf, wav_dict, title):
        # we will have keep extending this replacement list based on errors during compilation
        # escaping underscores in the title
        wav_buttons = []
        for key in wav_dict.keys():
            name = re.sub('_','',key)
            wav_buttons.append("\movie[label="+ name +"]{}{"+os.path.basename(wav_dict[key])+"}")
            wav_buttons.append("\hyperlinkmovie[play]{" + name + "}{\\beamerbutton{" + name + "}}")
        wav_buttons = "\n".join(wav_buttons)

        title = "\\texttt{"+re.sub("_","\_", title)+"}"
        fig_latex = """
%...

\\begin{frame}
""" + wav_buttons + """

\\begin{figure}[t]
  \\begin{center}
    \caption{\\texttt{""" + title + """ }}
    \includegraphics[height=0.5\\textheight]{""" + figure_pdf + """}
  \end{center}
\end{figure}
\clearpage
%
\end{frame}
%...
"""
        self.document.append(fig_latex)

    def Close(self):
        self.document.append("\end{document}")
        return self.Compile()

    def Compile(self):
        root, ext = os.path.splitext(self.pdf_file)
        dir_name = os.path.dirname(self.pdf_file)
        latex_file = root + ".tex"
        lat_file = open(latex_file, "w")
        lat_file.write("\n".join(self.document))
        lat_file.close()
        logger.info("Compiling the latex report.")
        try:
            print " ".join(['pdflatex', '-interaction=batchmode', '-output-directory='+str(dir_name), latex_file])
            proc = subprocess.Popen(['pdflatex', '-interaction=batchmode', '-output-directory='+str(dir_name), latex_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.communicate()
        except Exception as e:
            logger.warning("There was an error compiling the latex file {0}, please do it manually.".format(latex_file))
            return False
        return True

def ParseAlignmentFile(file_name, utt_names = None):
    #sp0.9-AMI_EN2001a_SDM_FEO065_0021133_0021442 57 5 ; 30 7 ; 45 5 ; 71 15 ; 135 12 ; 87 12 ; 98 22 ; 1 11 ; 161 8 ; 67 12 ; 54 6 ; 85 7 ; 107 10 ; 127 6 ; 35 10 ; 131 5 ; 143 10 ; 31 3 ; 111 5 ; 134 8 ; 85 6 ; 170 9 ; 33 10 ; 74 3 ; 1 3 ; 101 13 ; 43 7 ; 110 7 ; 1 3 ; 61 4 ; 170 20 ; 1 4 ; 161 3 ; 63 3 ; 102 12 ; 1 55
    logger.info("Parsing alignment file {0}".format(file_name))
    if utt_names is not None:
        utt_names = set(utt_names)

    alignments = {}
    for line in open(file_name,'r'):
        phones = line.split(";")
        is_first = True
        alignment = []
        for phone in phones:
            parts = phone.split()
            if is_first:
                assert(len(parts) == 3)
                utt_name = parts[0]
                if utt_names is not None:
                    if not utt_name in utt_names:
                        break
                alignment.append(parts[1:])
                is_first = False
            else:

                if(len(parts) != 2):
                    print phones
                    assert(1==0)
                alignment.append(parts)
        if len(alignment) > 0:
            alignments[utt_name] = alignment
    return alignments

def MakeContiguousAlignment(alignment, phone_mapper):
    contiguous_alignment = []
    for phone in alignment:
        for i in range(int(phone[1])):
            contiguous_alignment.append(phone_mapper(int(phone[0])))
    return contiguous_alignment

def GenerateUtterancePlotsFeat(ali_dict, output_dir, utt_name, feat_dict, wav_dict, text = None, latex_report = None, phone_list = None, phone_mapper = None):
    fig = plt.figure()
    keys = feat_dict.keys()
    keys.sort()
    num_sub_plots = len(keys) + 1
    assert(num_sub_plots<=9)
    xlims = []
    ax1 = None

    # find the image ranges
    f_max = sys.float_info.min
    f_min = sys.float_info.max

    for key in keys:
        f_max = max(numpy.amax(feat_dict[key]), f_max)
        f_min = min(numpy.amin(feat_dict[key]), f_min)

    for i in range(len(keys)):
        key = keys[i]
        plot_number = int('{0}1{1}'.format(num_sub_plots, i+1))

        if ax1 is not None:
            plt.subplot(plot_number, sharex = ax1)
        else:
            ax1 = plt.subplot(plot_number)

        plt.ylabel(key+"\nFilterbank Index")
        #plt.title(key)
        plt.imshow(numpy.transpose(feat_dict[key]), aspect = 'auto', origin = 'lower', vmax=f_max, vmin=f_min)
        plt.ylim(0, feat_dict[key].shape[1]-1)
        xlims = [0, feat_dict[key].shape[0]-1]

    plot_number = int('{0}1{1}'.format(num_sub_plots, num_sub_plots))
    ax2 = plt.subplot(plot_number, sharex = ax1)
    plots = []
    keys = ali_dict.keys()
    keys.sort()

    for i in range(len(keys)):
        color_val = plot_colors[i]
        array = MakeContiguousAlignment(ali_dict[keys[i]], phone_mapper)
        plot_handle, = plt.plot(array, color = color_val, linestyle = "-", label = " {0} ".format(keys[i]))
        plots.append(plot_handle)
    green_plot = [[1, 170 ] , [ 81, 5 ] , [ 154, 11 ] , [ 85, 8 ] , [ 170, 8 ] , [ 137 ,35 ] , [ 90 ,16 ] , [ 165, 6 ] , [ 151, 6 ] , [ 130, 5 ] , [ 141, 6 ] , [ 35 ,8 ] , [ 99, 3 ] , [ 87 ,3 ] , [ 114, 3 ] , [ 29, 9 ] , [ 47, 6 ] , [ 39, 12 ] , [ 142, 6 ] , [ 1, 35]]
    green_plot = MakeContiguousAlignment(green_plot, phone_mapper)
    green_plot = green_plot[16:]
    plot_handle, = plt.plot(green_plot, color = 'green' , linestyle = "-", label = " {0} ".format('IHM2'))
    plots.append(plot_handle)
    plt.xlabel('Frame Index')
    plt.ylabel('Phone Index')
    if phone_list is not None:
        plt.yticks(range(len(phone_list)), phone_list, fontsize = 1)
    #lgd = plt.legend(handles=plots, loc='lower center', bbox_to_anchor=(0.5, -0.2 + len(keys) * -0.3 ), ncol=1, borderaxespad=0.)
    lgd = plt.legend(handles=plots, loc=9, bbox_to_anchor=(0.5, -0.4  ), ncol=3, borderaxespad=0.)
    plt.grid(True)
    plt.xlim(xlims[0],xlims[1])
    figfile_name = '{0}/{1}.pdf'.format(output_dir, re.sub("\.", "_", utt_name))
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig(figfile_name, bbox_inches='tight')
    if latex_report is not None:
        latex_report.AddFigure(figfile_name, wav_dict, "Alignment plots for utterance {0} : {1} ".format(utt_name, text if text is not None else ''))
    plt.close(fig)

def GenerateUtterancePlots(ali_dict, output_dir, utt_name, text = None, latex_report = None):
    fig = plt.figure()
    plots = []

    keys = ali_dict.keys()
    keys.sort()

    #AMI_TS3009d_SDM_MTD033PM_0199199_0199414
    #AMI_TS3009d_SDM_MTD034ID_0199041_0199404
    for i in range(len(keys)):
        color_val = plot_colors[i]
        array = MakeContiguousAlignment(ali_dict[keys[i]])
        plot_handle, = plt.plot(array, color = color_val, linestyle = "-", label = " {0} ".format(keys[i]))
        plots.append(plot_handle)


    plt.xlabel('Frame Index')
    plt.ylabel('Phone Index')
    lgd = plt.legend(handles=plots, loc='lower center', bbox_to_anchor=(0.5, -0.2 + len(keys) * -0.1), ncol=2, mode="expand", borderaxespad=0.)
    plt.grid(True)
    figfile_name = '{0}/{1}.pdf'.format(output_dir, re.sub("\.", "_", utt_name))
    plt.savefig(figfile_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    if latex_report is not None:
        latex_report.AddFigure(figfile_name, "Alignment plots for utterance {0} : {1} ".format(utt_name, text if text is not None else ''))
    plt.close(fig)

def ParseTextFile(text_file):
    lines = open(text_file, 'r').readlines()
    text_dict = {}
    for line in lines:
        parts = line.split()
        text_dict[parts[0]] = " ".join(parts[1:])
    return text_dict

def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    logger.info("Running the command\n{0}".format(command))
    p = subprocess.Popen(command, shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception("There was an error while running the command {0}\n".format(command)+"-"*10+"\n"+stderr)
        return stdout, stderr
    else:
        return p

def LoadFeatures(utt_names, wav_dir, output_dir):
    segments = '{0}/segments'.format(wav_dir)
    if os.path.exists(segments):
        filtered_segments = []
        segments = open(segments,'r').readlines()
        for segment in segments:
            parts = segment.split()
            if parts[0] in utt_names:
                filtered_segments.append(segment)
        filt_segment_filename = '{0}/filtered_segments'.format(output_dir)
        filt_segment_file = open(filt_segment_filename, 'w')
        filt_segment_file.write('\n'.join(filtered_segments))
        filt_segment_file.close()
        segments = filt_segment_filename
    else:
        segments = ''

    RunKaldiCommand('extract-segments scp:{wdir}/wav.scp {segments} ark:-| compute-fbank-feats ark:- ark,t:{odir}/feats.txt'.format(wdir = wav_dir, odir = output_dir, segments = segments))
    import kaldi_io as kio
    ark_handle = kio.read_mat_ark('{odir}/feats.txt'.format(odir = output_dir))
    feat_dict = {}
    for key, mat in ark_handle:
        feat_dict[key] = mat

    return feat_dict

def ExtractWavFiles(utt_names, wav_dir, uniq_key, output_dir):
    segments = '{0}/segments'.format(wav_dir)
    filtered_segments = None
    if os.path.exists(segments):
        filtered_segments = {}
        # parse required segments
        segments = open(segments,'r').readlines()
        for segment in segments:
            parts = segment.split()
            if parts[0] in utt_names:
                filtered_segments[parts[0]] = [parts[1], float(parts[2]), float(parts[3])]

    # read wav_file
    wavs = {}
    for line in open('{0}/wav.scp'.format(wav_dir), 'r').readlines():
        parts = line.split()
        wav_command = ' '.join(parts[1:]).strip()
        if wav_command[-1] == '|':
            wav_command = wav_command[:-1]
        wavs[parts[0]] = wav_command

    import errno

    try:
        os.makedirs('{out}/wavs'.format(out=output_dir))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
            pass

    out_wavs = {}
    for utt_name in utt_names:
        out_file = '{out}/{utt}_{uk}.wav'.format(out = output_dir,
                                                 utt = utt_name.strip(),
                                                 uk = uniq_key)
        out_wavs[utt_name] = out_file
        if filtered_segments is not None:
            wav_name = filtered_segments[utt_name][0]
            wav_command = wavs[wav_name]
            wav_command = "{wc} | sox -t wav - {out_file} trim {start} {end}".format(
                    wc = wav_command,
                    out_file = out_file,
                    start = filtered_segments[utt_name][1],
                    end = filtered_segments[utt_name][2] - filtered_segments[utt_name][1])
        else:
            wav_command = wavs[utt_name]
            wav_command = "{wc} | sox -t wav - {out_file}".format(
                    wc = wav_command,
                    out_file = out_file)
        RunKaldiCommand(wav_command)

    return out_wavs

def PhoneMapper(phone_index, phone_list, unique_phone_list):
    phone_string = phone_list[phone_index]
    new_phone_index = [x for x in range(len(unique_phone_list)) if unique_phone_list[x] == phone_string]
    return new_phone_index[0]

def GeneratePlots(ali_wav, output_dir,
                  max_plots = None, text_file = None,
                  phone_file = None,
                  utt_name_file = None):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise e

    latex_report = LatexReport("{0}/alignments.pdf".format(output_dir))
    text_dict = None
    if text_file is not None:
        text_dict = ParseTextFile(text_file)

    if phone_file is not None:
        phone_lines = open(phone_file, 'r').readlines()
        phone_list = [None]*len(phone_lines)
        unique_phone_list = []
        old_phone = None
        for phone_line in phone_lines:
            parts = phone_line.split()
            phone = parts[0].split('_')[0]
            phone_index = int(parts[1])
            phone_list[phone_index] = phone
            if old_phone != phone:
                unique_phone_list.append(phone)
                old_phone = phone
        print unique_phone_list
        phone_mapper = lambda phone_index : PhoneMapper(phone_index, phone_list, unique_phone_list)
    else:
        phone_mapper = lambda x: x
        unique_phone_list = None

    utt_names = None
    if utt_name_file is not None:
        utt_names = map( lambda x: x.strip(), open(utt_name_file, 'r').readlines())

    parsed_alignments = {}
    feats = {}
    wavs = {}
    keys = ali_wav.keys()
    for  key in keys:
        ali_file_name = ali_wav[key]['ali_file']
        wav_dir = ali_wav[key]['wav_dir']
        parsed_alignments[key] = ParseAlignmentFile(ali_file_name, utt_names = utt_names)
        if utt_names is None:
            utt_names = parsed_alignments[key].keys()
            if max_plots is not None:
                shuffle(utt_names)
                utt_names = utt_names[:max_plots]

        if wav_dir is not None:
            logger.info("Extracting wavs for {0}.".format(key))
            wavs[key] = ExtractWavFiles(utt_names, wav_dir, key, output_dir)
            logger.info("Loading features for {0}.".format(key))
            feats[key] = LoadFeatures(utt_names, wav_dir, output_dir)

    logger.info("Loaded alignments, wavs and features (if specified)")

    key_list = parsed_alignments.keys()
    num_plots = 1
    for utt_name in utt_names:
        ali_dict = {}
        feat_dict = {}
        wav_dict = {}

        try:
            for key in key_list:
                ali_dict[key] = parsed_alignments[key][utt_name]
                feat_dict[key] = feats[key][utt_name] if len(feats.keys()) > 0 else None
                wav_dict[key] = wavs[key][utt_name] if len(wavs.keys()) > 0 else None
        except KeyError:
            continue

        if len(feats.keys()) == 0:
            GenerateUtterancePlots(ali_dict, output_dir, utt_name,
                                  text = text_dict[utt_name] if text_dict is not None else None,
                                  latex_report = latex_report)
        else:
            GenerateUtterancePlotsFeat(ali_dict, output_dir, utt_name, feat_dict, wav_dict,
                                  text = text_dict[utt_name] if text_dict is not None else None,
                                  latex_report = latex_report,
                                  phone_list = unique_phone_list,
                                  phone_mapper = phone_mapper)
        num_plots += 1
        if max_plots is not None and num_plots > max_plots:
            break

    has_compiled = latex_report.Close()
    if has_compiled:
        logger.info("Report has been generated. You can find it at the location {0}".format("{0}/alignments.pdf".format(output_dir)))

def Main():
    args = GetArgs()
    GeneratePlots(args.ali_wav, args.output_dir, max_plots = args.max_plots,
                                              text_file = args.text_file,
                                              phone_file = args.phone_file,
                                              utt_name_file = args.utt_name_file)

if __name__ == "__main__":
    print sys.argv[1:]
    Main()
