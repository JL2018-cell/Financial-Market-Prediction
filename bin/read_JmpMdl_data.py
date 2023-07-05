def read_JmpMdl(filename):
    # Reads the definition of parameters and returns a list of values.
    # The returned dict contains 
    # following information: {parameter_name: [parameter value]}
    parameters = dict()
    f = open(filename)
    for line in f:
        line_s  = line.split('\n')    
        line_spl  = line_s[0].split(' ')
        parameters[line_spl[0]] = float(line_spl[1])
    return parameters
