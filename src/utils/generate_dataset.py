# NOTICE: on ubuntu/linux please install : "apt-get install libarchive-dev" and then  use "pip3 install libarchive"

# OLD CODE!!!

if __name__ == "__main__":
    #OLD CODE TO REMOVE 
    ORDER_EVERY_TU = 15 # parametro da variare
    ORDERS_DURATION_TU = 15
    N_ORDERS_TU = 5 # parametro da variare
    N_LEVELS = 10
    POLLUTION_TYPE = PollutionType.BOTH_SELL_BUY 
    PERCENTAGES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] # desidered

    ORDER_EVERY_TU_ITER = [calculate_perturbation_frequence(p, ORDERS_DURATION_TU, N_ORDERS_TU, N_LEVELS) for p in PERCENTAGES]
    print(ORDER_EVERY_TU_ITER)

    N_ORDERS_TU_ITER = [calculate_perturbation_levels(p, ORDERS_DURATION_TU, ORDER_EVERY_TU, N_LEVELS) for p in PERCENTAGES]
    print(N_ORDERS_TU_ITER)

    # Example on how to load the data from lobster data dir
    load_data_from_lobster = False
    if load_data_from_lobster:
        k = 100
        out_df = from_folder_to_unique_df("AMZN_Sample", plot=False, level=10)
        out_df = add_lob_labels(out_df, rolling_tu=k, sign_threshold=0.002)
        # sample f
        f = calculate_perturbation_frequence(0.1, ORDERS_DURATION_TU, N_ORDERS_TU, N_LEVELS)
        out_df = pollute_training_df(df=out_df, order_every_tu=f, orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=N_ORDERS_TU, pollution_type=POLLUTION_TYPE, n_levels=N_LEVELS, seed=42) 

    # Load several data from F1 dataset in txt format and covert them to our format
    ids = ["7", "8", "9"]
    for t in ids:
        load_data_from_f1 = True
        file_data = 'data/Test_Dst_NoAuction_DecPre_CF_' + t + '.txt'
        if load_data_from_f1:
            initial_df = f1_file_dataset_to_lob_df(file_data)

        pollute_data = True
        if pollute_data:
            for f in ORDER_EVERY_TU_ITER:
                out_df = initial_df.copy()
                out_df = pollute_training_df(df=out_df, order_every_tu=f, orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=N_ORDERS_TU, pollution_type=POLLUTION_TYPE, n_levels=N_LEVELS, seed=SEED) 

                save_file_txt = True
                out_file = 'data/Adv_Test_Dst_NoAuction_DecPre_CF_' + str(f) + '_' + str(N_ORDERS_TU) + '_seed' + str(SEED) + '_F_' + t + '.txt'
                if save_file_txt:
                    lob_df_to_f1_dataset_file(out_df, out_file) 

            for l in N_ORDERS_TU_ITER:
                l = int(l)
                out_df = initial_df.copy()
                out_df = pollute_training_df(df=out_df, order_every_tu=ORDER_EVERY_TU, orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=l, pollution_type=POLLUTION_TYPE, n_levels=N_LEVELS, seed=SEED) 
            
                save_file_txt = True
                out_file = 'data/Adv_Test_Dst_NoAuction_DecPre_CF_' + str(ORDER_EVERY_TU) + '_' + str(l) + '_seed' + str(SEED) + '_L_' + t + '.txt'
                if save_file_txt:
                    lob_df_to_f1_dataset_file(out_df, out_file)
