weight=(45 41 27 26 26 25 21 17 17 14 13 13 12 12 12 11 10 10 10 10 9 9 9 9 8 8 7 7 7 7 6 6 6 6 5 4 4 4 4 4 4 4 4 4 3 3 3 3 3 3 3) 
w=(1 2 3 4 5)
echo "${w}"
echo {1..10}

python shoe_game.py \
--env non_symmetric_vote \
--N 51 \
--reward_list [0]*500+[1]*500 \
--weight $weight
--m 1000
