CREATE TABLE [dbo].[Department] (
    [Id] INT IDENTITY (1, 1) NOT NULL,
    [DepartmentName] NVARCHAR (50) NOT NULL, 
    [CreatedBy]    INT      NULL,
    [CreatedOn]    DATETIME NULL,
    [ModifiedBy]   INT      NULL,
    [ModifiedOn]   DATETIME NULL
);