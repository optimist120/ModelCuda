clear all
fileID = fopen("C:\Users\misha\source\repos\ThreeCuda\DataE.txt",'r');
A = fscanf(fileID,'%f');
B = reshape(A,102400,1);
figure
for i=1:1
R= histogram(B(:,i));
set(gca,'YScale','log','XScale','log')
% xlim([-1000 10000]);
% ylim([0 10000]);
% R.Visible = 'off';
% H = R.Values;
% loglog(H,'.r')
pause(0.2)

end

