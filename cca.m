function score = cca(X_file, Y_file)

X = dlmread(X_file);
Y = dlmread(Y_file);

X = normr(X);
Y = normr(Y);

[Wx,Wx,r] = canoncorr(X, Y);

score = mean(r);

fprintf('QVEC score: %f\n',score);
exit

