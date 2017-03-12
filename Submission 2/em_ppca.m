    filename = 'data.dat';
    T = importdata(filename);
    q = 2;
    plot(T);
    [W, sigma] = em_ppca_from_data(T, q);
    X = ppca_latent(T, W, sigma);
    figure;
    length(X)
    plot(X);
