#word = input("What would you like to eat?")
#if(word == "starter"):
#    print ("""

#We have the following: 
#cheese sticks
#sliders
#chicken fingers""")
#elif(word == "entree"):
#    print ("""
# Pizza
# Steak
# Sandwich""")
#else:
#    print ("Ok, no food then.")


#Functions: modify menu program to call a function for every option rather than simply calling print
#Lists: modify your menu program to use a list to store all of the options 
# rather than had-coding them into the if statements

#def goodans():
 #   print("good!")




#menu = ["cheese sticks", "sliders", "chicken fingers", "Pizza", "Steak", "Sandwich"]
#print (menu)

#def waiter():
    #food = input ("What would you like to eat? We have "cheese sticks", "sliders", "chicken fingers", "Pizza", "Steak", "Sandwich". Please tell me your starter and entree.")
    #waiter("cheese sticks", "Pizza")
    #if(food==menu(0) and (food==menu(3))):
    #    print("Excellent choice. Coming right up!")
   #     goodans()
  #  else:
 #       print("Your taste buds are kinda wack.")

#waiter()

#def func():
    # food = ["cheese sticks", "sliders", "chicken fingers", "Pizza", "Steak", "Sandwich"]
 #   pass;


#def menu(food):
 #   print(food);
    
  #  if(food == "Pizza"):
   #     func()
    #    print("Excellence choice. Coming right up!")
    #else:
     #   func()
      #  print("Your taste buds are kinda wack.")



#waiter = input("What would you like to eat?")
#print(waiter);
#menu(waiter)
#menu("Pizza")

# def goodAns():
#     print("Good choice")

# def badAns():
#     print("no")

# options = ["pizza", "bread", "worms"]

# inputStr = "What would you like to eat?"

def goodAns():
    print("Wow! You have the finest of taste ;)")

def badAns():
    print("Your taste buds are kinda wack.")

food = ["Pizza", "Sausage", "Hamburger", "Cheese Sticks"]

while(True):
    answer = input("What would you like to eat?: " + str(food) + "\n")
    print("\n")

    if (answer == food[0]):
        goodAns()
    elif(answer == food[1]):
        badAns()
    else:
        print("Never eat again bud.")

print("\n\n\n")

anything