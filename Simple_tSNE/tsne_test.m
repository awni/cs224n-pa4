addpath('/Users/awni/Desktop/cs224n/homework/pa4/Simple_tSNE');

%% Load and process data

%wordVecs = load('wordVectors.txt');

wordVecs = load('word_vecs_final.txt');
wordVecs = wordVecs';

load indices;
indices = indices(1:10000);
wordVecs = wordVecs(indices+1,:);




%% Train

% Set parameters
no_dims = 2;
init_dims = 30;
perplexity = 30;

% Run t_SNE
mappedX = tsne(wordVecs, [], no_dims, init_dims, perplexity);
save 'mapped_L_final'

%% Plot

load mapped_L_final;
words = importdata('word_by_count.txt', ' ');
%words = words(1:10000);
toPlot = 1000;
a=figure('Color', 'w');     
scatter(mappedX(1:toPlot,1), mappedX(1:toPlot,2), 0.001); axis off
h = text(mappedX(1:toPlot,1), mappedX(1:toPlot,2), words(1:toPlot));
set(h, 'FontSize', 9);
print(a,'-djpeg','-r800','finalsneorig')

oldscreenunits = get(gcf,'Units');
oldpaperunits = get(gcf,'PaperUnits');
oldpaperpos = get(gcf,'PaperPosition');
set(gcf,'Units','pixels');
scrpos = get(gcf,'Position');
newpos = scrpos/100;
set(gcf,'PaperUnits','inches',...
'PaperPosition',newpos)
print('-dpng', 'finalsne2', '-r300');
drawnow
set(gcf,'Units',oldscreenunits,...
'PaperUnits',oldpaperunits,...
'PaperPosition',oldpaperpos)
%%Preprocessing already done
% load indices;
% words = importdata('final_train_words.txt', ' ');
% wordsbycount = importdata('word_by_count.txt', ' ');
% 
% 
% wordsMapped = [];
% wordsIndex = [];
% 
% for i=1:numel(wordsbycount)
%    ind = strmatch(wordsbycount(i), words, 'exact');
%    if mod(i,1000)==0
%        disp(i)
%    end
%    if length(ind)>0
%      wordsMapped = [wordsMapped;wordsbycount(i)];
%      wordsIndex = [wordsIndex; indices(ind)];
%    end
% end
