import mylib

### test 1:
print("Test 1: 'word'")
print("Expected: ['w', 'wo', 'or', 'rd', 'd']")
print(f"Got: {mylib.calc_char_bigrams('word')}")

### test 2:
print("Test 2: 'a b c'")
print("Expected: ['a', 'a', 'b', 'b', 'c', 'c']")
print(f"Got: {mylib.calc_char_bigrams('a b c')}")

### test 3:
print("Test 3: 'Hi!     How are you?'")
print("Expected: ['H', 'Hi', 'i', 'H', 'Ho', 'ow', 'w', 'a', 'ar', 're', 'e', 'y', 'yo', 'ou', 'u']")
print(f"Got: {mylib.calc_char_bigrams('Hi!     How are you?')}")

### test 4:
print("Test 4: 'THIS    @is A.weird!!&TEST       String'")
print("Expected: ['T', 'TH', 'HI', 'IS', 'S', 'i', 'is', 's', 'A', 'A', 'w', 'we', 'ei', 'ir', 'rd', 'd', 'T', 'TE', 'ES', 'ST', 'T', 'S', 'St', 'tr', 'ri', 'in', 'ng', 'g']")
print(f"Got: {mylib.calc_char_bigrams('THIS    @is A.weird!!&TEST       String')}")