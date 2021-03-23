ist_l = 'Interstitial glucose'
cho_l = 'Carbohydrate intake'
phy_l = 'Physical activity'
inb_l = 'requested insulin bolus'
inr_l = 'requested insulin basal rate'


WINDOW_WIDTH_1H = 12    #1 hour window
WINDOW_WIDTH_24H = 288  #24 hour window

def print_h(text):
    print('\n' + '+' * (len(text)+6))
    print(f'++ {text} ++')
    print('+' * (len(text)+6) + '\n')

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '=', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '>' + '.' * (length - filledLength)
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end = printEnd)