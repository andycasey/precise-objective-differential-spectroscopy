import moog

def test_moog():

    with moog.instance("/tmp") as moogsilent:
        None
