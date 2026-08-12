CREATE TABLE [dbo].[BeneficiaryType] (
    [Id] INT IDENTITY (1, 1) NOT NULL,
    [Name] NVARCHAR (50) NOT NULL, 
    [CreatedOn] DATETIME NOT NULL DEFAULT GETDATE(),
);