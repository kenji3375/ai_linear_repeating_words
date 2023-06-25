import random

min_len = 5
max_len = 32

number_of_strings = 7000


number_of_strings = int(number_of_strings)


data = []


letter_tokens = [
    "",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l","m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v","w", "x", "y", "z", "0", "1", "2", "3",
    "4", "5","6", "7", "8", "9", "!", "@", "#", "$",
    "%", "^","&", "*", "(", ")", "-", "_", "+", "=",
    "[", "]", "{", "}", "|", "\\", ";", ":", "'", '"',
    ",", ".", "/", "<", ">", "?", "`", "~", " ",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L","M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V","W", "X", "Y", "Z"
]



data = []

for i in range(number_of_strings):
    random_string = ""
    #select the length of the password
    length = random.randint(min_len, max_len)
    for x in range(length):
        #select the letter
        letter = random.choice(letter_tokens)
        random_string += letter
    data.append(random_string)

#let's tokenize the data and data



#tokenize data




data2 = []


for item in data:
    #print(item)
    tok_arr = []
    for letter in range(0, max_len):
        if letter < len(item):
            tok_arr.append(letter_tokens.index(item[letter]))
        else:
            tok_arr.append(0)
    data2.append(tok_arr)

data = data2


#print(data)

final_data = []


for item in data:
    final_data.append([item, item])
    #print(item)

data = final_data