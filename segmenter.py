

def morph_segmenter(model):
  return lambda username: GetSegments(model, username)


def GetSegments(model, username):
  segments, _ = model.viterbi_segment(username)
  segments = [s.decode('utf8') for s in segments]

  return segments


def baseline_segmenter(username, nchars=4):
  name = u'#' + username.decode('utf8') + u'#'
  z = zip(*[name[i:] for i in range(nchars)])
  segments = [u''.join(seg) for seg in z]

  return segments
