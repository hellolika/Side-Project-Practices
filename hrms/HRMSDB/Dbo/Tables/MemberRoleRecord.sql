CREATE TABLE [dbo].[MemberRoleRecord] (
    [Id]         INT      IDENTITY (1, 1) NOT NULL,
    [MemberId]   INT      NOT NULL,
    [RoleId]     INT      NOT NULL,
    [CreatedBy]  INT      NULL,
    [CreatedOn]  DATETIME NOT NULL,
    [ModifiedBy] INT      NULL,
    [ModifiedOn] DATETIME NOT NULL
);
GO

