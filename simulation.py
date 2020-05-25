
def run_simul(real, predicted, initial = 2500, verbose = True):
    assert (len(real) == len(predicted))
    buys = [[],[]]
    big_buys = [[],[]]
    sells = [[],[]]
    spent, num_shares, revenue = 0, 0, 0
    max_spent, temp_spent = 0, 0
    total_loss = 0
    limit = initial
    for i in range(len(predicted)-1):
        change = (predicted[i+1] - predicted[i]) / (predicted[i])
        if change > 0.005 and int((limit - temp_spent) / real[i]) > 0:
            order_size = 1
            if change > 0.01:
                order_size = min(int((limit - temp_spent) / real[i]), 15)  # use ALL remaining money to buy
                if verbose:
                    print("MAJOR BUY AT: " + str(real[i]) + ", CHANGE: " + str(change))
                big_buys[0].append(i)
                big_buys[1].append(real[i])
            else:
                if verbose:
                    print("BOUGHT AT: " + str(real[i]) + ", CHANGE: " + str(change))
                buys[0].append(i)
                buys[1].append(real[i])
            spent += order_size * real[i]
            temp_spent += order_size * real[i]
            num_shares += order_size

        if change < -0.005 and num_shares > 0:
            revenue += num_shares * real[i]
            num_shares = 0
            if temp_spent > max_spent:
                max_spent = temp_spent
            temp_spent = 0

            if limit > initial + (revenue - spent):
                loss = limit - (initial + (revenue - spent))
                total_loss += loss
                if verbose:
                    print("LOST: " + str(loss))

            limit = initial + (revenue - spent)     # can spend everything it earns
            if verbose:
                print("LIMIT: " + str(limit))
            sells[0].append(i)
            sells[1].append(real[i])
            if verbose:
                print("SOLD AT: " + str(real[i]))

    print("TOTAL LOSS: " + str(total_loss))
    print("MAX SPENT: " + str(max_spent))
    print("NUM SHARES: " + str(num_shares) + ", PRICED AT: " + str(real[len(real)-2]))
    print("Profited: " + str(revenue - spent))
    return revenue - spent, buys, sells, big_buys
