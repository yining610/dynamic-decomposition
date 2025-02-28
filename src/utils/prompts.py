FACTSCORE_VERIFY_PROMPT = """
Your task is to verify the correctness of the given claim. Only answer with 'True' or 'False'.

Input: In 1996, he was awarded the Ig Nobel Prize for Art, an award given to honor achievements that make people laugh, and then think. True or False?
Output: True
Input: Featherstone's pink flamingo design was displayed at the Smithsonian National Museum of American History in 1996, and he was inducted into the Plastics Hall of Fame in 1998. True or False?
Output: False
Input: Featherstone continued to work on his designs until his death in 2015, and his creations remain popular among collectors and enthusiasts of Americana. True or False?
Output: False
Input: Travis Oliphant is a data scientist and entrepreneur who is best known for creating the NumPy and SciPy libraries for Python programming language. True or False?
Output: True
Input: He was born on August 22, 1972, in the United States. True or False?
Output: False
"""

FACTSCORE_DECOMPOSITION_PROMPT = {
    "system": "Seperate the decomposed subclaims with a hyphen.",
    "prompt":
"""
Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to
appear in small and supporting roles throughout the 1990s.
- He made his acting debut in the film.
- He made his acting debut in The Moon is the Sun's Dream.
- The Moon is the Sun's Dream is a film.
- The Moon is the Sun's Dream was released in 1992.
- After his acting debut, he appeared in small and supporting roles.
- After his acting debut, he appeared in small and supporting roles throughout the 1990s.

Please breakdown the following sentence into independent facts: He is also a successful producer and engineer, having worked with a wide variety of artists,
including Willie Nelson, Tim McGraw, and Taylor Swift.
- He is successful.
- He is a producer.
- He is a engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.

Please breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served
as the back-up Command Module Pilot for the Gemini 7 mission.
- Collins became an astronaut.
- Collins became one of the third group of astronauts.
- Collins became one of the third group of astronauts selected.
- Collins became one of the third group of astronauts selected by NASA.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- He served as the Command Module Pilot.
- He served as the back-up Command Module Pilot.
- He served as the Command Module Pilot for the Gemini 7 mission.

Please breakdown the following sentence into independent facts: In addition to his acting roles, Bateman has written and directed two short films and is
currently in development on his feature debut.
- Bateman has acting roles.
- Bateman has written two short films.
- Bateman has directed two short films.
- Bateman has written and directed two short films.
- Bateman is currently in development on his feature debut.

Please breakdown the following sentence into independent facts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who
was the Command Module Pilot for the Apollo 11 mission in 1969.
- Michael Collins was born on October 31, 1930.
- Michael Collins is retired.
- Michael Collins is an American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins was the Command Module Pilot.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission.
- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Please breakdown the following sentence into independent facts: He was an American composer, conductor, and musical director.
- He was an American.
- He was a composer.
- He was a conductor.
- He was a musical director.

Please breakdown the following sentence into independent facts: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.
- She currently stars in Love and Destiny.
- Love and Destiny is a romantic comedy series.
- Love and Destiny premiered in 2019.

Please breakdown the following sentence into independent facts: During his professional career, McCoy played for the Broncos, the San Diego Chargers, the
Minnesota Vikings, and the Jacksonville Jaguars.
- McCoy played for the Broncos.
- McCoy played for the Broncos during his professional career.
- McCoy played for the San Diego Chargers.
- McCoy played for the San Diego Chargers during his professional career.
- McCoy played for the Minnesota Vikings.
- McCoy played for the Minnesota Vikings during his professional career.
- McCoy played for the Jacksonville Jaguars.
- McCoy played for the Jacksonville Jaguars during his professional career.

Please breakdown the following sentence into independent facts: {claim}
"""
}

# Annotated FactScore Prompts for more atmoic decomposition
FACTSCORE_ATOM_DECOMPOSITION_PROMPT = {
    "system": "You are a decomposer. Your task is to decompose the given claim into more granular subclaims. There are two principles you have to follow: 1) making sure there is no information loss or gain after decomposition and 2) making sure each generated subclaim is self-contained. Seperate the decomposed subclaims with a hyphen.",
    "prompt":
"""
Following the given two principles, please decompose the following claim into more granular subclaims: He made his acting debut in the film The Moon is the Sun's Dream.
- He made his acting debut.
- Debut happened in the film.
- The Moon is the Sun's Dream is a film.

Following the given two principles, please decompose the following claim into more granular subclaims: He has worked with a wide variety of artists.
- He worked.
- It happened with a wide variety of artists.

Following the given two principles, please decompose the following claim into more granular subclaims: Bateman has directed two short films.
- Bateman had directed films.
- There are two films.
- Films are short.

Following the given two principles, please decompose the following claim into more granular subclaims: {claim}
"""
}

BINARY_SPLIT_DECOMPOSITION_PROMPT = {
    "system": "You are a decomposer. Your task is to decompose the given claim into two sub-claims. There are two principles you have to follow: 1) making sure there is no information loss or gain after decomposition and 2) making sure each generated subclaim is self-contained and approximately equal in length and information. Seperate the two subclaims with a hyphen.",
    "prompt": 
"""
Following the given two principles, please decompose the following claim into two sub-claims: In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission.
- Collins became one of the third group of astronauts selected by NASA in 1963.
- Collins served as the back-up Command Module Pilot for the Gemini 7 mission.

Following the given two principles, please decompose the following claim into two sub-claims: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.
- In addition to his acting roles, Bateman has written and directed two short films.
- Bateman is currently in development on his feature debut.

Following the given two principles, please decompose the following claim into two sub-claims: "Parasite" received widespread critical acclaim for its screenplay, direction, acting, and its social commentary.
- "Parasite" received widespread critical acclaim for its screenplay and direction.
- "Parasite" received widespread critical acclaim for its acting and social commentary.

Following the given two principles, please decompose the following claim into two sub-claims: {claim}
"""
}

WICE_PROMPT = {
    "system": "Seperate the decomposed subclaims with a hyphen.",
    "prompt":
"""
Segment the following sentence into individual facts:

Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Facts:
- Lord Steven Regal wan the World Television Championship. 
- The Nasty Boys wan and the World Tag Team Championship.

Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

Sentence: Touring began in Europe in April-June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July-August, and concluding in North America in October-November.
Facts:
- Touring began in Europe in April-June.
- The opening act was guitarist Paul Gilbert.
- There was a tour in Australia in July.
- There was a tour in New Zealand in July.
- There was a tour in Mexico in late July-August.
- There was a tour in South America in late July-August
- The tour was concluded in North America in October-November.

Sentence: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Facts:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The two companies partnered to offer Al-enabled conversational solutions to customers in India.

Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Facts:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- Yazidi community in Germany lives primarily in Hannover.
- Yazidi community in Germany lives primarily in Bielefeld.
- Yazidi community in Germany lives primarily in Celle.
- Yazidi community in Germany lives primarily in Bremen.
- Yazidi community in Germany lives primarily in Bad Oeynhausen.
- Yazidi community in Germany lives primarily in Pforzheim.
- Yazidi community in Germany lives primarily in Oldenburg.

Sentence: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.
Facts:
- Sebastian Vettel is a previous six-time winner of the Nations' Cup.
- Sebastian Vettel became Champion of Champions for the first time.
- Sebastian Vettel defeated Tom Kristensen.
- Tom Kristensen made the final for the fourth time.
- The score was 2-0.

Sentence: {claim}
Facts:\n"""
}


RND_PROMPT = {
    "system": "Seperate the decomposed subclaims with a hyphen.",
    "prompt":
"""
Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to
appear in small and supporting roles throughout the 1990s.
- He has an acting debut.
- He acted in a film.
- His acting debut was in a film.
- His acting debut was in The Moon is the Sun's Dream.
- He acted in The Moon is the Sun's Dream.
- The Moon is the Sun's Dream is a film.
- The Moon is the Sun's Dream was released in 1992.
- His acting debut occurred in 1992.
- He appeared in small roles.
- He appeared in supporting roles.
- His small roles occurred throughout the 1990s.
- His supporting roles occurred throughout the 1990s.
- His appearance in small roles occurred after his acting debut.
- His appearance in supporting roles occurred after his acting debut.

Please breakdown the following sentence into independent facts: He is also a successful producer and engineer, having worked with a wide variety of artists,
including Willie Nelson, Tim McGraw, and Taylor Swift.
- He is a producer.
- He is successful at being a producer.
- He is an engineer.
- He is successful at being an engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.

Please breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served
as the back-up Command Module Pilot for the Gemini 7 mission.
- NASA selected a third group of astronauts.
- Collins belonged to the third group of astronauts.
- Collins was selected by NASA.
- Collins's selection by NASA occurred in 1963.
- The Gemini 7 mission has a back-up Command Module Pilot.
- Collins's role in the Gemini 7 mission was as the back-up Command Module Pilot.
- Collins participated in the Gemini 7 mission.

Please breakdown the following sentence into independent facts: In addition to his acting roles, Bateman has written and directed two short films and is
currently in development on his feature debut.
- Bateman has acting roles.
- Bateman has written short films.
- The number of short films Bateman has written is two.
- Bateman has directed short films.
- The number of short films Bateman has directed is two.
- Bateman is currently in development on his feature debut.
- The two short films were made before his feature debut.
- His acting roles came before his feature debut.

Please breakdown the following sentence into independent facts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who
was the Command Module Pilot for the Apollo 11 mission in 1969.
- Michael Collins was born in October.
- Michael Collins was born on the 31st day of a month.
- Michael Collins was born in 1930.
- Michael Collins is retired.
- Michael Collins is American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins participated in the Apollo 11 mission.
- Michael Collins's participation in the Apollo 11 mission occurred in 1969.
- The Apollo 11 mission was active in 1969.
- The day of Michael Collins's birth occurred before his year of participation in the Apollo 11 mission.
- The Apollo 11 mission had a Command Module Pilot.
- Michael Collins's role in the Apollo 11 mission was as the Command Module Pilot.

Please breakdown the following sentence into independent facts: He was an American composer, conductor, and musical director.
- He was American.
- He was a composer.
- He was a conductor.
- He was a musical director.

Please breakdown the following sentence into independent facts: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.
- She stars in Love and Destiny.
- Love and Destiny is a series.
- Love and Destiny is a romantic comedy.
- Love and Destiny premiered in 2019.

Please breakdown the following sentence into independent facts: During his professional career, McCoy played for the Broncos, the San Diego Chargers, the
Minnesota Vikings, and the Jacksonville Jaguars.
- McCoy had a professional career.
- McCoy played for the Broncos.
- McCoy played for the San Diego Chargers.
- The Chargers are from San Diego.
- McCoy played for the Minnesota Vikings.
- The Vikings are from Minnesota.
- McCoy played for the Jacksonville Jaguars.
- The Jaguars are from Jacksonville.

Please breakdown the following sentence into independent facts: {claim}
"""
}