
def GetFrameShift(data_dir):
    frame_shift, = RunKaldiCommand("utils/data/get_frame_shift.sh {0}".format(data_dir))
    return int(frame_shift)

def GenerateUtt2Dur(data_dir):
    RunKaldiCommand("utils/data/get_utt2dur.sh {0}".format(data_dir))

def GetUtt2Dur(data_dir):
    GenerateUtt2Dur(data_dir)
    utt2dur = {}
    for line in open('{0}/utt2dur'.format(data_dir), 'r').readlines():
        parts = line.split()
        utt2dur[parts[0]] = float(parts[1])
    return utt2dur

def GetUtt2Uniq(data_dir):
    utt2uniq_file = '{0}/utt2uniq'.format(data_dir)
    if not os.path.exists(utt2uniq):
        return None, None
    utt2uniq = {}
    uniq2utt = {}
    for line in open(utt2uniq_file, 'r').readlines():
        parts = line.split()
        utt2uniq[parts[0]] = parts[1]
        if uniq2utt.has_key(parts[1]):
            uniq2utt[parts[1]].append(parts[0])
        else:
            uniq2utt[parts[1]] = [parts[0]]
    return utt2uniq, uniq2utt

def GetNumFrames(data_dir):
    GenerateUtt2Dur(data_dir)
    frame_shift = GetFrameShift(data_dir)
    total_duration = 0
    for item in GetUtt2Dur(data_dir).items():
        total_duration = total_duration + item[1]
    return int(float(total_duration)/frame_shift)

