from string import punctuation

def clean_string(string):
    new_string = ""
    for _ in string:
      asc = ord(_)
      if asc!=12 and asc!=7 and asc < 128:
          if _ not in punctuation:
              new_string += _
          else:
              new_string += " "
    return new_string
