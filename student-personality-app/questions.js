// Big Five Personality Questions
// Based on standard Big Five/OCEAN personality assessment items
// 10 questions per trait, some reverse-scored

const personalityQuestions = [
    // OPENNESS TO EXPERIENCE (O)
    {
        id: 1,
        trait: 'O',
        text: 'I have a vivid imagination.',
        reversed: false
    },
    {
        id: 2,
        trait: 'O',
        text: 'I am interested in abstract ideas.',
        reversed: false
    },
    {
        id: 3,
        trait: 'O',
        text: 'I enjoy thinking about theoretical concepts.',
        reversed: false
    },
    {
        id: 4,
        trait: 'O',
        text: 'I prefer routine and familiar experiences.',
        reversed: true
    },
    {
        id: 5,
        trait: 'O',
        text: 'I enjoy trying new foods and activities.',
        reversed: false
    },
    {
        id: 6,
        trait: 'O',
        text: 'I appreciate art, music, and literature.',
        reversed: false
    },
    {
        id: 7,
        trait: 'O',
        text: 'I like to explore new ideas and ways of thinking.',
        reversed: false
    },
    {
        id: 8,
        trait: 'O',
        text: 'I have difficulty understanding abstract concepts.',
        reversed: true
    },
    {
        id: 9,
        trait: 'O',
        text: 'I am curious about many different things.',
        reversed: false
    },
    {
        id: 10,
        trait: 'O',
        text: 'I prefer to stick with things I know.',
        reversed: true
    },

    // CONSCIENTIOUSNESS (C)
    {
        id: 11,
        trait: 'C',
        text: 'I am always prepared for class.',
        reversed: false
    },
    {
        id: 12,
        trait: 'C',
        text: 'I pay attention to details in my work.',
        reversed: false
    },
    {
        id: 13,
        trait: 'C',
        text: 'I get my homework done right away.',
        reversed: false
    },
    {
        id: 14,
        trait: 'C',
        text: 'I often forget where I put things.',
        reversed: true
    },
    {
        id: 15,
        trait: 'C',
        text: 'I like to have a schedule and stick to it.',
        reversed: false
    },
    {
        id: 16,
        trait: 'C',
        text: 'I make plans and follow through with them.',
        reversed: false
    },
    {
        id: 17,
        trait: 'C',
        text: 'I leave my belongings around.',
        reversed: true
    },
    {
        id: 18,
        trait: 'C',
        text: 'I am careful to do things correctly.',
        reversed: false
    },
    {
        id: 19,
        trait: 'C',
        text: 'I sometimes make a mess of things.',
        reversed: true
    },
    {
        id: 20,
        trait: 'C',
        text: 'I work hard to achieve my goals.',
        reversed: false
    },

    // EXTRAVERSION (E)
    {
        id: 21,
        trait: 'E',
        text: 'I feel comfortable around people.',
        reversed: false
    },
    {
        id: 22,
        trait: 'E',
        text: 'I start conversations easily.',
        reversed: false
    },
    {
        id: 23,
        trait: 'E',
        text: 'I am the life of the party.',
        reversed: false
    },
    {
        id: 24,
        trait: 'E',
        text: 'I prefer to keep to myself.',
        reversed: true
    },
    {
        id: 25,
        trait: 'E',
        text: 'I talk to a lot of different people at social events.',
        reversed: false
    },
    {
        id: 26,
        trait: 'E',
        text: 'I enjoy being the center of attention.',
        reversed: false
    },
    {
        id: 27,
        trait: 'E',
        text: 'I don\'t talk a lot.',
        reversed: true
    },
    {
        id: 28,
        trait: 'E',
        text: 'I have a lot of energy and enthusiasm.',
        reversed: false
    },
    {
        id: 29,
        trait: 'E',
        text: 'I prefer quiet, one-on-one conversations.',
        reversed: true
    },
    {
        id: 30,
        trait: 'E',
        text: 'I am talkative and expressive.',
        reversed: false
    },

    // AGREEABLENESS (A)
    {
        id: 31,
        trait: 'A',
        text: 'I am interested in other people\'s problems.',
        reversed: false
    },
    {
        id: 32,
        trait: 'A',
        text: 'I feel others\' emotions.',
        reversed: false
    },
    {
        id: 33,
        trait: 'A',
        text: 'I make people feel at ease.',
        reversed: false
    },
    {
        id: 34,
        trait: 'A',
        text: 'I am not really interested in others.',
        reversed: true
    },
    {
        id: 35,
        trait: 'A',
        text: 'I take time out for others.',
        reversed: false
    },
    {
        id: 36,
        trait: 'A',
        text: 'I cooperate well with others.',
        reversed: false
    },
    {
        id: 37,
        trait: 'A',
        text: 'I can be cold and uncaring sometimes.',
        reversed: true
    },
    {
        id: 38,
        trait: 'A',
        text: 'I trust what others tell me.',
        reversed: false
    },
    {
        id: 39,
        trait: 'A',
        text: 'I believe that people are basically good.',
        reversed: false
    },
    {
        id: 40,
        trait: 'A',
        text: 'I sometimes find it hard to see things from another person\'s viewpoint.',
        reversed: true
    },

    // NEUROTICISM (N)
    {
        id: 41,
        trait: 'N',
        text: 'I get stressed out easily.',
        reversed: false
    },
    {
        id: 42,
        trait: 'N',
        text: 'I worry about things.',
        reversed: false
    },
    {
        id: 43,
        trait: 'N',
        text: 'I am easily disturbed.',
        reversed: false
    },
    {
        id: 44,
        trait: 'N',
        text: 'I am relaxed most of the time.',
        reversed: true
    },
    {
        id: 45,
        trait: 'N',
        text: 'I get upset easily.',
        reversed: false
    },
    {
        id: 46,
        trait: 'N',
        text: 'I change my mood a lot.',
        reversed: false
    },
    {
        id: 47,
        trait: 'N',
        text: 'I rarely feel blue or sad.',
        reversed: true
    },
    {
        id: 48,
        trait: 'N',
        text: 'I often feel anxious about things.',
        reversed: false
    },
    {
        id: 49,
        trait: 'N',
        text: 'I seldom get mad.',
        reversed: true
    },
    {
        id: 50,
        trait: 'N',
        text: 'I sometimes feel overwhelmed by my emotions.',
        reversed: false
    }
];

// Trait information for results
const traitInfo = {
    O: {
        name: 'Openness to Experience',
        shortName: 'Openness',
        color: '#f56565',
        high: 'You are highly open to new experiences! You tend to be creative, curious, and imaginative. You enjoy exploring new ideas, art, and ways of thinking. You\'re likely drawn to learning new things and appreciate intellectual discussions.',
        medium: 'You have a balanced approach to new experiences. You can appreciate both familiar routines and trying new things. You\'re open to learning but also value what you already know.',
        low: 'You prefer familiar experiences and practical thinking. You value tradition and concrete ideas over abstract concepts. You\'re grounded and prefer sticking with what works for you.'
    },
    C: {
        name: 'Conscientiousness',
        shortName: 'Conscientiousness',
        color: '#48bb78',
        high: 'You are highly conscientious! You\'re organized, responsible, and goal-oriented. You plan ahead, pay attention to details, and follow through on commitments. You\'re likely a reliable student who gets work done on time.',
        medium: 'You have a balanced level of conscientiousness. You can be organized when needed but are also flexible. You get things done but don\'t stress too much about perfect planning.',
        low: 'You prefer to be spontaneous and flexible rather than strictly planned. You might sometimes procrastinate or be disorganized, but you\'re also adaptable and creative in how you approach tasks.'
    },
    E: {
        name: 'Extraversion',
        shortName: 'Extraversion',
        color: '#ed8936',
        high: 'You are highly extraverted! You gain energy from being around people and enjoy social situations. You\'re likely talkative, enthusiastic, and comfortable in group settings. You probably have many friends and enjoy meeting new people.',
        medium: 'You have a balanced level of extraversion. You can enjoy both social situations and time alone. You\'re comfortable in groups but also appreciate quiet time to recharge.',
        low: 'You are more introverted and gain energy from time alone or with small groups. You prefer deep conversations over small talk and may need time to recharge after social events. You\'re likely thoughtful and reflective.'
    },
    A: {
        name: 'Agreeableness',
        shortName: 'Agreeableness',
        color: '#4299e1',
        high: 'You are highly agreeable! You\'re compassionate, cooperative, and care about others\' feelings. You tend to be trusting, helpful, and good at working in teams. People likely see you as kind and considerate.',
        medium: 'You have a balanced level of agreeableness. You can be cooperative and helpful while also standing up for yourself when needed. You balance empathy with healthy skepticism.',
        low: 'You are more independent in your thinking and less influenced by others\' opinions. You\'re analytical, competitive, and willing to challenge ideas. You prioritize logic over emotions in decision-making.'
    },
    N: {
        name: 'Neuroticism',
        shortName: 'Neuroticism',
        color: '#9f7aea',
        high: 'You score high on neuroticism, meaning you experience emotions intensely. You may be more sensitive to stress and experience mood swings. This isn\'t bad - it means you\'re emotionally aware and may be more empathetic. Consider practicing stress management techniques.',
        medium: 'You have moderate emotional stability. You experience normal ranges of emotions and can usually manage stress effectively. You have some emotional ups and downs but generally cope well.',
        low: 'You are emotionally stable and resilient. You tend to stay calm under pressure and don\'t get upset easily. You\'re even-tempered and rarely feel overwhelmed by negative emotions.'
    }
};

// Shuffle array function for randomizing questions
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}
