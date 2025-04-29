
create database fake_indian_currency;
use fake_indian_currency;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    email VARCHAR(50), 
    password VARCHAR(50)
    );
