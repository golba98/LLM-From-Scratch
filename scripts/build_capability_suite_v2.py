#!/usr/bin/env python3
"""Build the frozen v2 foundation capability suite (32 new gold items per category).

  .venv/bin/python scripts/build_capability_suite_v2.py   # writes src/codexa_workspace/benchmarks/foundation_capability_v2.json

The v1 suite (56 gold + 8 conversational items, ``foundation_capability_v1.json``) is preserved
unchanged; v2 adds new, disjoint items in the same seven categories. Every item exists in a
plain-completion form (scored with a leading space, the pretraining format) and a chat form
(scored without a leading space, as the SFT serializer encodes assistant text). Grammar items
carry their two options so the evaluator can present them in both orders (choice-order control).
Gold answers were written before any model output was seen and are never edited to match one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'src/codexa_workspace/benchmarks/foundation_capability_v2.json'
ONE_WORD = ' Reply with only one word.'
NUMBER = ' Reply with only the number.'
NAME = ' Reply with only the name.'

FACTUAL = [
    ('The capital of Germany is', 'What is the capital of Germany?', 'Berlin', ['Munich', 'Vienna', 'Hamburg', 'Paris']),
    ('The capital of Spain is', 'What is the capital of Spain?', 'Madrid', ['Barcelona', 'Lisbon', 'Seville', 'Rome']),
    ('The capital of Canada is', 'What is the capital of Canada?', 'Ottawa', ['Toronto', 'Vancouver', 'Montreal', 'Quebec']),
    ('The capital of Australia is', 'What is the capital of Australia?', 'Canberra', ['Sydney', 'Melbourne', 'Perth', 'Brisbane']),
    ('The capital of Egypt is', 'What is the capital of Egypt?', 'Cairo', ['Alexandria', 'Giza', 'Luxor', 'Tunis']),
    ('The capital of Russia is', 'What is the capital of Russia?', 'Moscow', ['Kiev', 'Minsk', 'Warsaw', 'Prague']),
    ('The capital of China is', 'What is the capital of China?', 'Beijing', ['Shanghai', 'Tokyo', 'Seoul', 'Taipei']),
    ('The capital of Greece is', 'What is the capital of Greece?', 'Athens', ['Sparta', 'Rome', 'Sofia', 'Istanbul']),
    ('The chemical symbol for gold is', 'What is the chemical symbol for gold?', 'Au', ['Ag', 'Fe', 'Go', 'Gd']),
    ('The chemical symbol for sodium is', 'What is the chemical symbol for sodium?', 'Na', ['So', 'Sd', 'K', 'Ne']),
    ('The planet closest to the Sun is', 'Which planet is closest to the Sun?', 'Mercury', ['Venus', 'Mars', 'Earth', 'Jupiter']),
    ('The largest ocean on Earth is the', 'What is the largest ocean on Earth?', 'Pacific', ['Atlantic', 'Indian', 'Arctic', 'Southern']),
    ('The longest river in Africa is the', 'What is the longest river in Africa?', 'Nile', ['Congo', 'Niger', 'Zambezi', 'Amazon']),
    ('Romeo and Juliet was written by William', 'Who wrote Romeo and Juliet? Reply with only the surname.', 'Shakespeare', ['Wordsworth', 'Blake', 'Dickens', 'Marlowe']),
    ('The first person to walk on the Moon was Neil', 'Who was the first person to walk on the Moon? Reply with only the surname.', 'Armstrong', ['Aldrin', 'Collins', 'Gagarin', 'Glenn']),
    ('At sea level, water boils at 100 degrees', 'At sea level, water boils at 100 degrees on which temperature scale?', 'Celsius', ['Fahrenheit', 'Kelvin', 'Rankine', 'Newton']),
    ('The hardest natural substance is', 'What is the hardest natural substance?', 'diamond', ['gold', 'iron', 'granite', 'quartz']),
    ('Plants take in carbon', 'Plants take in carbon what from the air?', 'dioxide', ['monoxide', 'oxide', 'dust', 'fibre']),
    ('The currency of Japan is the', 'What is the currency of Japan?', 'yen', ['won', 'yuan', 'dollar', 'euro']),
    ('The currency of the United States is the', 'What is the currency of the United States?', 'dollar', ['euro', 'pound', 'peso', 'yen']),
    ('The largest animal on Earth is the blue', 'The largest animal on Earth is the blue what?', 'whale', ['shark', 'bear', 'elephant', 'dolphin']),
    ('The fastest land animal is the', 'What is the fastest land animal?', 'cheetah', ['lion', 'horse', 'tiger', 'gazelle']),
    ('Bees make', 'What food do bees make?', 'honey', ['milk', 'silk', 'bread', 'wool']),
    ('The organ that pumps blood around the body is the', 'Which organ pumps blood around the body?', 'heart', ['liver', 'lung', 'brain', 'kidney']),
    ('The number of continents on Earth is', 'How many continents are there on Earth? Reply with only the number word.', 'seven', ['five', 'six', 'eight', 'nine']),
    ('The number of legs on a spider is', 'How many legs does a spider have? Reply with only the number word.', 'eight', ['six', 'four', 'ten', 'twelve']),
    ('The main language spoken in Brazil is', 'What is the main language spoken in Brazil?', 'Portuguese', ['Spanish', 'French', 'Italian', 'English']),
    ('The tallest mountain in the world is Mount', 'What is the tallest mountain in the world? Reply with only the mountain name.', 'Everest', ['Kilimanjaro', 'Fuji', 'Elbrus', 'Denali']),
    ('The Great Pyramid of Giza is in', 'In which country is the Great Pyramid of Giza?', 'Egypt', ['Mexico', 'Greece', 'Peru', 'India']),
    ('The Colosseum is in the city of', 'In which city is the Colosseum?', 'Rome', ['Athens', 'Paris', 'Venice', 'Naples']),
    ('The opposite of north is', 'What is the opposite of north?', 'south', ['east', 'west', 'up', 'down']),
    ('The season after winter is', 'Which season comes after winter?', 'spring', ['summer', 'autumn', 'fall', 'monsoon']),
]

ARITHMETIC = [
    ('1 + 5 =', 'What is 1 + 5?', '6', ['5', '7', '4', '15']),
    ('4 + 3 =', 'What is 4 + 3?', '7', ['6', '8', '1', '43']),
    ('8 + 1 =', 'What is 8 + 1?', '9', ['8', '10', '7', '81']),
    ('2 + 6 =', 'What is 2 + 6?', '8', ['7', '9', '4', '26']),
    ('5 + 5 =', 'What is 5 + 5?', '10', ['9', '11', '55', '25']),
    ('7 + 3 =', 'What is 7 + 3?', '10', ['9', '11', '4', '73']),
    ('6 + 6 =', 'What is 6 + 6?', '12', ['11', '13', '36', '66']),
    ('9 + 2 =', 'What is 9 + 2?', '11', ['10', '12', '7', '92']),
    ('10 + 5 =', 'What is 10 + 5?', '15', ['14', '16', '105', '50']),
    ('12 + 3 =', 'What is 12 + 3?', '15', ['14', '16', '9', '36']),
    ('20 + 10 =', 'What is 20 + 10?', '30', ['20', '40', '200', '31']),
    ('8 - 3 =', 'What is 8 - 3?', '5', ['4', '6', '11', '83']),
    ('9 - 4 =', 'What is 9 - 4?', '5', ['4', '6', '13', '94']),
    ('7 - 2 =', 'What is 7 - 2?', '5', ['4', '6', '9', '72']),
    ('10 - 3 =', 'What is 10 - 3?', '7', ['6', '8', '13', '30']),
    ('6 - 1 =', 'What is 6 - 1?', '5', ['4', '6', '7', '61']),
    ('15 - 5 =', 'What is 15 - 5?', '10', ['9', '11', '20', '155']),
    ('12 - 4 =', 'What is 12 - 4?', '8', ['7', '9', '16', '3']),
    ('3 times 2 is', 'What is 3 times 2?', '6', ['5', '8', '9', '32']),
    ('4 times 2 is', 'What is 4 times 2?', '8', ['6', '9', '16', '42']),
    ('5 times 2 is', 'What is 5 times 2?', '10', ['7', '12', '25', '52']),
    ('3 times 3 is', 'What is 3 times 3?', '9', ['6', '8', '12', '33']),
    ('2 times 5 is', 'What is 2 times 5?', '10', ['7', '12', '20', '25']),
    ('10 times 2 is', 'What is 10 times 2?', '20', ['12', '30', '102', '10']),
    ('6 divided by 2 is', 'What is 6 divided by 2?', '3', ['2', '4', '8', '12']),
    ('8 divided by 2 is', 'What is 8 divided by 2?', '4', ['2', '6', '10', '16']),
    ('10 divided by 5 is', 'What is 10 divided by 5?', '2', ['5', '3', '15', '50']),
    ('12 divided by 3 is', 'What is 12 divided by 3?', '4', ['3', '6', '9', '36']),
    ('Half of 10 is', 'What is half of 10?', '5', ['2', '10', '20', '4']),
    ('Double 4 is', 'What is double 4?', '8', ['6', '2', '16', '44']),
    ('One more than 9 is', 'What number is one more than 9?', '10', ['8', '11', '19', '91']),
    ('One less than 7 is', 'What number is one less than 7?', '6', ['5', '8', '17', '71']),
]

COPY_EXTRACT = [
    ('Color: green. The color is', 'Color: green. What is the color?', 'green', ['red', 'blue', 'yellow', 'white']),
    ('The code word is falcon. The code word is', 'The code word is falcon. What is the code word?', 'falcon', ['eagle', 'hawk', 'raven', 'robin']),
    ("Name: Omar. City: Cairo. The person's name is", "Name: Omar. City: Cairo. What is the person's name?", 'Omar', ['Cairo', 'Ali', 'Hassan', 'Karim']),
    ('Name: Omar. City: Cairo. The city is', 'Name: Omar. City: Cairo. What is the city?', 'Cairo', ['Omar', 'Giza', 'Tunis', 'Rabat']),
    ('Ticket number: 47. The ticket number is', 'Ticket number: 47. What is the ticket number?', '47', ['74', '40', '7', '48']),
    ('The box contains apples. The box contains', 'The box contains apples. What does the box contain?', 'apples', ['pears', 'plums', 'grapes', 'lemons']),
    ('Pet: rabbit. Owner: Lucy. The pet is a', 'Pet: rabbit. Owner: Lucy. What is the pet?', 'rabbit', ['cat', 'dog', 'Lucy', 'hamster']),
    ('Pet: rabbit. Owner: Lucy. The owner is', 'Pet: rabbit. Owner: Lucy. Who is the owner?', 'Lucy', ['Lucas', 'Mary', 'rabbit', 'Linda']),
    ('Room 12 is on the second floor. The room number is', 'Room 12 is on the second floor. What is the room number?', '12', ['21', '2', '10', '22']),
    ('The train leaves at 9. The train leaves at', 'The train leaves at 9. At what time does the train leave?', '9', ['8', '10', '6', '19']),
    ('Word: river. The word above is', 'The word is river. What is the word?', 'river', ['lake', 'stream', 'ocean', 'bridge']),
    ('First item: pen. Second item: cup. The second item is', 'First item: pen. Second item: cup. What is the second item?', 'cup', ['pen', 'bowl', 'mug', 'plate']),
    ('First item: pen. Second item: cup. The first item is', 'First item: pen. Second item: cup. What is the first item?', 'pen', ['cup', 'pencil', 'book', 'ink']),
    ('Fruit: mango. Vegetable: carrot. The vegetable is', 'Fruit: mango. Vegetable: carrot. What is the vegetable?', 'carrot', ['mango', 'potato', 'onion', 'pea']),
    ('Fruit: mango. Vegetable: carrot. The fruit is', 'Fruit: mango. Vegetable: carrot. What is the fruit?', 'mango', ['carrot', 'apple', 'banana', 'peach']),
    ('Animal: zebra. The animal is a', 'Animal: zebra. What is the animal?', 'zebra', ['horse', 'lion', 'giraffe', 'donkey']),
    ('The secret number is 83. The secret number is', 'The secret number is 83. What is the secret number?', '83', ['38', '80', '3', '84']),
    ("Tom's favourite colour is purple. Tom's favourite colour is", "Tom's favourite colour is purple. What is Tom's favourite colour?", 'purple', ['pink', 'blue', 'orange', 'brown']),
    ('Sara lives in Lima. Sara lives in', 'Sara lives in Lima. Where does Sara live?', 'Lima', ['Lisbon', 'Lyon', 'Quito', 'Bogota']),
    ('The key is under the mat. The key is under the', 'The key is under the mat. What is the key under?', 'mat', ['door', 'rug', 'bed', 'table']),
    ('Language: Swahili. The language is', 'Language: Swahili. What is the language?', 'Swahili', ['Spanish', 'Somali', 'Zulu', 'Arabic']),
    ('Score: 15. The score is', 'Score: 15. What is the score?', '15', ['51', '5', '10', '16']),
    ('Day: Wednesday. The day is', 'Day: Wednesday. What is the day?', 'Wednesday', ['Tuesday', 'Thursday', 'Monday', 'Friday']),
    ('Month: October. The month is', 'Month: October. What is the month?', 'October', ['September', 'November', 'August', 'December']),
    ('Instrument: violin. The instrument is a', 'Instrument: violin. What is the instrument?', 'violin', ['piano', 'cello', 'guitar', 'flute']),
    ('Planet: Neptune. The planet is', 'Planet: Neptune. What is the planet?', 'Neptune', ['Uranus', 'Saturn', 'Pluto', 'Jupiter']),
    ('Tool: hammer. The tool is a', 'Tool: hammer. What is the tool?', 'hammer', ['saw', 'drill', 'nail', 'spanner']),
    ('Drink: coffee. Food: bread. The drink is', 'Drink: coffee. Food: bread. What is the drink?', 'coffee', ['bread', 'tea', 'juice', 'milk']),
    ('Drink: coffee. Food: bread. The food is', 'Drink: coffee. Food: bread. What is the food?', 'bread', ['coffee', 'rice', 'cake', 'cheese']),
    ('Sport: tennis. The sport is', 'Sport: tennis. What is the sport?', 'tennis', ['football', 'golf', 'cricket', 'hockey']),
    ('Bird: parrot. The bird is a', 'Bird: parrot. What is the bird?', 'parrot', ['pigeon', 'crow', 'eagle', 'owl']),
    ('Metal: copper. The metal is', 'Metal: copper. What is the metal?', 'copper', ['iron', 'silver', 'gold', 'tin']),
]

SENTENCE_COMPLETION = [
    ('The grass is', 'green', ['blue', 'red', 'loud', 'hot']),
    ('Snow is white and coal is', 'black', ['white', 'green', 'sweet', 'loud']),
    ('Dogs bark and cats', 'meow', ['bark', 'fly', 'swim', 'sing']),
    ('At night we sleep in a', 'bed', ['car', 'tree', 'river', 'cup']),
    ('You eat soup with a', 'spoon', ['fork', 'knife', 'pen', 'comb']),
    ('The sun is very', 'hot', ['cold', 'wet', 'small', 'quiet']),
    ('An apple a day keeps the doctor', 'away', ['home', 'near', 'busy', 'happy']),
    ('Rain falls from the', 'sky', ['ground', 'sea', 'tree', 'floor']),
    ('Birds have feathers and fish have', 'scales', ['feathers', 'wings', 'fur', 'hair']),
    ('Monday comes before', 'Tuesday', ['Sunday', 'Friday', 'Monday', 'Saturday']),
    ('January is the first month of the', 'year', ['week', 'day', 'decade', 'hour']),
    ('A baby cat is called a', 'kitten', ['puppy', 'calf', 'foal', 'cub']),
    ('A baby dog is called a', 'puppy', ['kitten', 'calf', 'lamb', 'chick']),
    ('The opposite of up is', 'down', ['up', 'left', 'over', 'high']),
    ('The opposite of big is', 'small', ['large', 'tall', 'huge', 'wide']),
    ('A bicycle has two', 'wheels', ['doors', 'wings', 'legs', 'eyes']),
    ('We hear with our', 'ears', ['eyes', 'hands', 'feet', 'nose']),
    ('Children go to school to', 'learn', ['sleep', 'swim', 'cook', 'drive']),
    ('Summer is hot and winter is', 'cold', ['hot', 'wet', 'long', 'green']),
    ('People wear shoes on their', 'feet', ['hands', 'heads', 'ears', 'knees']),
    ('A clock tells the', 'time', ['weather', 'story', 'news', 'truth']),
    ('Thank you very', 'much', ['many', 'more', 'well', 'good']),
    ('Once upon a', 'time', ['day', 'night', 'story', 'place']),
    ('Ladies and', 'gentlemen', ['men', 'boys', 'girls', 'ladies']),
    ('Good morning, how are', 'you', ['we', 'they', 'it', 'he']),
    ('A doctor works in a', 'hospital', ['school', 'farm', 'bank', 'kitchen']),
    ('A teacher works in a', 'school', ['hospital', 'farm', 'bank', 'garage']),
    ('Milk comes from', 'cows', ['chickens', 'trees', 'bees', 'fish']),
    ('The moon shines at', 'night', ['noon', 'breakfast', 'lunch', 'dawn']),
    ('Books are borrowed from a', 'library', ['kitchen', 'garage', 'bathroom', 'garden']),
    ('To open a locked door you need a', 'key', ['spoon', 'cup', 'book', 'hat']),
    ('A fish lives in', 'water', ['sand', 'trees', 'fire', 'air']),
]

GRAMMAR = [  # (sentence with ___, gold, wrong)
    ('He ___ a dog.', 'has', 'have'), ('We ___ ready.', 'are', 'is'),
    ('She ___ to the shop yesterday.', 'went', 'go'), ('The cat ___ on the mat every morning.', 'sits', 'sit'),
    ('I ___ happy today.', 'am', 'are'), ('They ___ playing football.', 'are', 'is'),
    ('My brother ___ tall.', 'is', 'are'), ('Last week we ___ a film.', 'watched', 'watch'),
    ('She ___ like coffee.', "doesn't", "don't"), ('There ___ a book on the table.', 'is', 'are'),
    ('The dogs ___ loudly every night.', 'bark', 'barks'), ('He ___ his homework every day.', 'does', 'do'),
    ('You ___ my friend.', 'are', 'is'), ('It ___ cold yesterday.', 'was', 'is'),
    ('Tomorrow we ___ visit Grandma.', 'will', 'did'), ('I have ___ apple.', 'an', 'a'),
    ('She is ___ teacher.', 'a', 'an'), ('This is ___ book.', 'my', 'me'),
    ('Give the pen to ___.', 'me', 'I'), ('___ is my sister.', 'She', 'Her'),
    ('The boys ___ already eaten lunch.', 'have', 'has'), ('Yesterday she ___ a letter.', 'wrote', 'write'),
    ('He can ___ very fast.', 'run', 'runs'), ('We ___ in London last year.', 'were', 'are'),
    ('Mary and John ___ friends.', 'are', 'is'), ('The sun ___ in the east.', 'rises', 'rise'),
    ('I ___ seen that film before.', 'have', 'has'), ('My shoes ___ new.', 'are', 'is'),
    ('Look at ___ birds over there.', 'those', 'that'), ('This box is ___ than that one.', 'heavier', 'heavy'),
    ('He is the ___ boy in the class.', 'tallest', 'tall'), ('They ___ to music every evening.', 'listen', 'listens'),
]

REASONING = [
    ('Ann is older than Ben. The younger one is', 'Ann is older than Ben. Who is younger?' + NAME, 'Ben', ['Ann']),
    ('Sam finished before Leo. The first to finish was', 'Sam finished before Leo. Who finished first?' + NAME, 'Sam', ['Leo']),
    ('The red box is heavier than the blue box. The lighter box is the', 'The red box is heavier than the blue box. Which box is lighter? Reply with only the colour.', 'blue', ['red']),
    ('Kim gave a pen to Raj. Now the pen belongs to', 'Kim gave a pen to Raj. Who has the pen now?' + NAME, 'Raj', ['Kim']),
    ('Today is Friday. Tomorrow will be', 'Today is Friday. What day is tomorrow?' + ONE_WORD, 'Saturday', ['Thursday', 'Sunday', 'Friday', 'Monday']),
    ('Yesterday was Monday. Today is', 'Yesterday was Monday. What day is today?' + ONE_WORD, 'Tuesday', ['Monday', 'Sunday', 'Wednesday', 'Friday']),
    ('I had 5 sweets and ate 2. Now I have', 'I had 5 sweets and ate 2. How many sweets do I have now?' + NUMBER, '3', ['2', '5', '7', '4']),
    ('There were 4 birds and 1 flew away. Now there are', 'There were 4 birds and 1 flew away. How many birds are left?' + NUMBER, '3', ['4', '5', '1', '2']),
    ('A cup is full of water. After you pour all the water out, the cup is', 'A cup is full of water. You pour all the water out. Is the cup full or empty?' + ONE_WORD, 'empty', ['full', 'heavy', 'hot', 'broken']),
    ('The lamp is on. After you switch it off, the lamp is', 'The lamp is on. You switch it off. Is the lamp on or off?' + ONE_WORD, 'off', ['on', 'bright', 'hot', 'new']),
    ('Mia has 2 cats and 1 dog. The number of pets Mia has is', 'Mia has 2 cats and 1 dog. How many pets does Mia have?' + NUMBER, '3', ['2', '1', '4', '5']),
    ('Tom is taller than Sue, and Sue is taller than Ali. The tallest is', 'Tom is taller than Sue, and Sue is taller than Ali. Who is the tallest?' + NAME, 'Tom', ['Sue', 'Ali']),
    ('Tom is taller than Sue, and Sue is taller than Ali. The shortest is', 'Tom is taller than Sue, and Sue is taller than Ali. Who is the shortest?' + NAME, 'Ali', ['Tom', 'Sue']),
    ('The shop opens at 9 in the morning and closes at 5 in the afternoon. At 7 in the evening the shop is', 'The shop opens at 9 in the morning and closes at 5 in the afternoon. Is the shop open or closed at 7 in the evening?' + ONE_WORD, 'closed', ['open', 'busy', 'new', 'full']),
    ('A glass falls on the floor and breaks. Now the glass is', 'A glass falls on the floor and breaks. What state is the glass in now?' + ONE_WORD, 'broken', ['whole', 'full', 'new', 'clean']),
    ("Ella has a red ball and Max has a green ball. Max's ball is", "Ella has a red ball and Max has a green ball. What colour is Max's ball?" + ONE_WORD, 'green', ['red']),
    ('Ella has a red ball and Max has a green ball. The red ball belongs to', 'Ella has a red ball and Max has a green ball. Who owns the red ball?' + NAME, 'Ella', ['Max']),
    ('The bus left at 3 and the train left at 4. The one that left first was the', 'The bus left at 3 and the train left at 4. Which left first, the bus or the train?' + ONE_WORD, 'bus', ['train']),
    ("Lily is Ben's sister. Ben is Lily's", "Lily is Ben's sister. What is Ben to Lily?" + ONE_WORD, 'brother', ['sister', 'father', 'mother', 'uncle']),
    ("Tom is Jack's father. Jack is Tom's", "Tom is Jack's father. What is Jack to Tom?" + ONE_WORD, 'son', ['father', 'brother', 'uncle', 'daughter']),
    ('Leo had 4 coins and found 1 more. Now Leo has', 'Leo had 4 coins and found 1 more. How many coins does Leo have?' + NUMBER, '5', ['4', '6', '3', '1']),
    ('Nina is in the kitchen and Paul is in the garden. The person in the garden is', 'Nina is in the kitchen and Paul is in the garden. Who is in the garden?' + NAME, 'Paul', ['Nina']),
    ('Nina is in the kitchen and Paul is in the garden. Nina is in the', 'Nina is in the kitchen and Paul is in the garden. Which room is Nina in?' + ONE_WORD, 'kitchen', ['garden', 'bedroom', 'garage', 'hall']),
    ('The cake was baked before the bread. The bread was baked', 'The cake was baked before the bread. Was the bread baked before or after the cake?' + ONE_WORD, 'after', ['before', 'during', 'never', 'again']),
    ('All birds have feathers. A robin is a bird, so a robin has', 'All birds have feathers. A robin is a bird. What does a robin have?' + ONE_WORD, 'feathers', ['fur', 'scales', 'fins', 'wheels']),
    ('If today is Sunday, then yesterday was', 'Today is Sunday. What day was yesterday?' + ONE_WORD, 'Saturday', ['Monday', 'Friday', 'Sunday', 'Tuesday']),
    ('Ben is 10 years old and Eva is 7. The older child is', 'Ben is 10 years old and Eva is 7. Who is older?' + NAME, 'Ben', ['Eva']),
    ('Ben is 10 years old and Eva is 7. The younger child is', 'Ben is 10 years old and Eva is 7. Who is younger?' + NAME, 'Eva', ['Ben']),
    ('The bottle is empty, so the bottle contains', 'The bottle is empty. What does the bottle contain?' + ONE_WORD, 'nothing', ['water', 'milk', 'juice', 'sand']),
    ('Ali arrived after Zoe. The last to arrive was', 'Ali arrived after Zoe. Who arrived last?' + NAME, 'Ali', ['Zoe']),
    ('A week has 7 days, so two weeks have', 'A week has 7 days. How many days are in two weeks?' + NUMBER, '14', ['7', '12', '21', '10']),
    ('Rosa is taller than Kai. The taller person is', 'Rosa is taller than Kai. Who is taller?' + NAME, 'Rosa', ['Kai']),
]

EXACT_FORMAT = [  # chat instruction, gold, distractors (surface variants included)
    ('Reply with only the word YES.', 'YES', ['Yes', 'yes', 'NO', 'OK']),
    ('Reply with only the word apple.', 'apple', ['Apple', 'APPLE', 'pear', 'fruit']),
    ('Reply with only the number 3.', '3', ['three', '4', '2', '30']),
    ('Reply with only the number 12.', '12', ['twelve', '21', '11', '13']),
    ('Write only the letter B.', 'B', ['b', 'A', 'C', 'Bee']),
    ('Write only the letter z.', 'z', ['Z', 'y', 'a', 'zed']),
    ('Answer yes or no: Is ice cold? No explanation.', 'yes', ['no', 'Yes', 'maybe', 'No']),
    ('Answer yes or no: Is the sun cold? No explanation.', 'no', ['yes', 'No', 'maybe', 'Yes']),
    ('Answer true or false: Two plus two is four.', 'true', ['false', 'True', 'yes', 'maybe']),
    ('Answer true or false: Snow is hot.', 'false', ['true', 'False', 'no', 'maybe']),
    ('Reply with only the word OK.', 'OK', ['ok', 'Ok', 'YES', 'Okay']),
    ('Reply with only the word green.', 'green', ['Green', 'GREEN', 'blue', 'red']),
    ('Reply with only the number 100.', '100', ['hundred', '10', '1000', '101']),
    ('Reply with only the word DONE.', 'DONE', ['Done', 'done', 'OK', 'READY']),
    ('Reply with only the word cat.', 'cat', ['Cat', 'CAT', 'dog', 'kitten']),
    ('Write only the digit 5.', '5', ['five', '6', '4', '55']),
    ('Answer with only the colour of a ripe banana, in lowercase.', 'yellow', ['Yellow', 'green', 'red', 'brown']),
    ('Answer with only the number of days in a week, as a digit.', '7', ['seven', '5', '6', '8']),
    ('Answer with only the first letter of the alphabet, as a capital letter.', 'A', ['a', 'B', 'Z', 'Alpha']),
    ('Answer with only the opposite of yes, in lowercase.', 'no', ['No', 'yes', 'maybe', 'not']),
    ('Reply with only the word hello in capital letters.', 'HELLO', ['Hello', 'hello', 'HI', 'HEY']),
    ('Reply with only the word goodbye in lowercase.', 'goodbye', ['Goodbye', 'GOODBYE', 'bye', 'hello']),
    ('Reply with only the number that comes after 4, as a digit.', '5', ['4', '6', 'five', '3']),
    ('Reply with only the word STOP in lowercase.', 'stop', ['STOP', 'Stop', 'go', 'halt']),
    ('Answer with one lowercase word: what colour is snow?', 'white', ['White', 'black', 'grey', 'blue']),
    ('Answer with one lowercase word: what animal says moo?', 'cow', ['Cow', 'dog', 'cat', 'sheep']),
    ('Answer with one lowercase word: what is frozen water called?', 'ice', ['Ice', 'snow', 'steam', 'water']),
    ('Reply with only the capital letter X.', 'X', ['x', 'Y', 'Z', 'ex']),
    ('Reply with only the digit 0.', '0', ['zero', '1', 'O', '10']),
    ('Reply with only the word maybe.', 'maybe', ['Maybe', 'MAYBE', 'yes', 'no']),
    ('Answer yes or no: Do fish live in water? No explanation.', 'yes', ['no', 'Yes', 'maybe', 'No']),
    ('Answer with only the number of legs a cat has, as a digit.', '4', ['four', '2', '3', '6']),
]

CONVERSATIONAL = ['Good evening!', 'Can you tell me a joke?', 'What is the weather usually like in summer?',
                  'I feel tired today.', 'Please introduce yourself.', 'What should I cook for dinner tonight?',
                  'Do you like music?', 'See you tomorrow!']


def build() -> dict:
    items = []

    def add(category, plain, chat, gold, distractors, **extra):
        items.append(dict(id=f'v2-{category}-{sum(i["category"] == category for i in items) + 1:02d}',
                          category=category, plain=plain, chat=chat, gold=gold, distractors=distractors, **extra))
    for plain, chat, gold, wrong in FACTUAL:
        add('factual', plain, chat if chat.endswith(('.', 'word.')) and 'Reply' in chat else chat + ONE_WORD, gold, wrong)
    for plain, chat, gold, wrong in ARITHMETIC:
        add('arithmetic', plain, chat + NUMBER, gold, wrong)
    for plain, chat, gold, wrong in COPY_EXTRACT:
        add('copy_extract', plain, chat + (NUMBER if gold.isdigit() else ONE_WORD), gold, wrong)
    for plain, gold, wrong in SENTENCE_COMPLETION:
        add('sentence_completion', plain, f'Complete the sentence: {plain} ...' + ONE_WORD, gold, wrong)
    for sentence, gold, wrong in GRAMMAR:
        add('grammar', f'{sentence} ({{options}}) The correct word is',
            f"Fill the gap: '{sentence}' Is it {{options_or}}? Reply with only the word.", gold, [wrong],
            options=[gold, wrong])
    for plain, chat, gold, wrong in REASONING:
        add('reasoning', plain, chat, gold, wrong)
    for chat, gold, wrong in EXACT_FORMAT:
        add('exact_format', f'Instruction: {chat}\nAnswer:', chat, gold, wrong)
    for index, question in enumerate(CONVERSATIONAL, start=1):
        items.append(dict(id=f'v2-conversational-{index:02d}', category='conversational', plain=None, chat=question,
                          gold=None, distractors=[]))
    content = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    return dict(format='codexa-foundation-capability-suite-v2', name='v2', content_sha256=content,
                disjoint_from='v1', review='authored by Claude Code 2026-10-13 before any v2 model output; '
                'automated validation only; independent human review PENDING', items=items)


if __name__ == '__main__':
    document = build()
    OUTPUT.write_text(json.dumps(document, indent=1, ensure_ascii=False) + '\n')
    print(OUTPUT, document['content_sha256'], len(document['items']))
